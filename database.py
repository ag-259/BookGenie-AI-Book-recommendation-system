import sqlite3
from contextlib import contextmanager
import bcrypt
from datetime import datetime
import re
import os

DATABASE_NAME = 'bookgenie.db'

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initialize the database with all required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                theme TEXT DEFAULT 'dark'
            )
        ''')
        
        # Liked books table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liked_books (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, book_id)
            )
        ''')
        
        # Reading progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reading_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('want_to_read', 'reading', 'completed')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, book_id)
            )
        ''')
        
        # Search history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                query TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        
        # Trending searches (global)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trending_searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                last_searched TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # RATINGS TABLE 
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, book_id)
            )
        ''')
        
        conn.commit()
        print("✅ Database initialized successfully!")

# ============== PASSWORD UTILITIES ==============

def hash_password(password):
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password, hashed_password):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# ============== USER OPERATIONS ==============

def create_user(username, email, password):
    """Create new user"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            # Validate email
            if not validate_email(email):
                return False, "Invalid email format"
            
            # Validate password
            is_valid, message = validate_password(password)
            if not is_valid:
                return False, message
            
            # Hash password
            hashed_pw = hash_password(password)
            
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, hashed_pw)
            )
            return True, cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return False, "Username already exists"
            elif 'email' in str(e):
                return False, "Email already registered"
            return False, "Registration failed"

def login_user(email, password):
    """Authenticate user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        
        if user and verify_password(password, user['password_hash']):
            return True, {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'theme': user['theme']
            }
        return False, None

def get_user_by_id(user_id):
    """Get user by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        return dict(user) if user else None

def update_user_theme(user_id, theme):
    """Update user's theme preference"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET theme = ? WHERE id = ?', (theme, user_id))
        return cursor.rowcount > 0

# ============== LIKED BOOKS ==============

def add_liked_book(user_id, book_id):
    """Add book to liked list"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO liked_books (user_id, book_id) VALUES (?, ?)',
                (user_id, book_id)
            )
            return True
        except sqlite3.IntegrityError:
            return False

def remove_liked_book(user_id, book_id):
    """Remove book from liked list"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM liked_books WHERE user_id = ? AND book_id = ?',
            (user_id, book_id)
        )
        return cursor.rowcount > 0

def get_liked_books(user_id):
    """Get all liked book IDs"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT book_id FROM liked_books WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        )
        return [row['book_id'] for row in cursor.fetchall()]

def is_book_liked(user_id, book_id):
    """Check if book is liked"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT 1 FROM liked_books WHERE user_id = ? AND book_id = ?',
            (user_id, book_id)
        )
        return cursor.fetchone() is not None

# ============== READING PROGRESS ==============

def set_reading_status(user_id, book_id, status):
    """Set reading status for a book"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''INSERT INTO reading_progress (user_id, book_id, status) 
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id, book_id) 
                   DO UPDATE SET status=?, updated_at=CURRENT_TIMESTAMP''',
                (user_id, book_id, status, status)
            )
            return True
        except Exception as e:
            print(f"Error setting reading status: {e}")
            return False

def get_reading_status(user_id, book_id):
    """Get reading status for a book"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT status FROM reading_progress WHERE user_id = ? AND book_id = ?',
            (user_id, book_id)
        )
        result = cursor.fetchone()
        return result['status'] if result else None

def get_books_by_status(user_id, status):
    """Get all books with specific status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT book_id FROM reading_progress WHERE user_id = ? AND status = ? ORDER BY updated_at DESC',
            (user_id, status)
        )
        return [row['book_id'] for row in cursor.fetchall()]

def remove_reading_status(user_id, book_id):
    """Remove reading status"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'DELETE FROM reading_progress WHERE user_id = ? AND book_id = ?',
            (user_id, book_id)
        )   
        return cursor.rowcount > 0

# ============== SEARCH HISTORY ==============

def add_search(user_id, query):
    """Add search to history"""
    with get_db() as conn:
        cursor = conn.cursor()
        # Add to user's search history
        if user_id:
            cursor.execute(
                'INSERT INTO search_history (user_id, query) VALUES (?, ?)',
                (user_id, query)
            )
        
        # Update trending searches
        cursor.execute(
            '''INSERT INTO trending_searches (query, count) 
               VALUES (?, 1)
               ON CONFLICT(query) 
               DO UPDATE SET count = count + 1, last_searched = CURRENT_TIMESTAMP''',
            (query,)
        )

# ============== COLLABORATIVE FILTERING ==============

def get_users_who_liked_book(book_id, exclude_user_id=None, limit=50):
    """Get users who liked a specific book"""
    with get_db() as conn:
        cursor = conn.cursor()
        if exclude_user_id:
            cursor.execute(
                'SELECT user_id FROM liked_books WHERE book_id = ? AND user_id != ? LIMIT ?',
                (book_id, exclude_user_id, limit)
            )
        else:
            cursor.execute(
                'SELECT user_id FROM liked_books WHERE book_id = ? LIMIT ?',
                (book_id, limit)
            )
        return [row['user_id'] for row in cursor.fetchall()]

def get_collaborative_recommendations(user_id, limit=20, min_likes=2):
    """Get book recommendations based on similar users (collaborative filtering)"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get books liked by current user
        cursor.execute('SELECT book_id FROM liked_books WHERE user_id = ?', (user_id,))
        user_liked_books = [row['book_id'] for row in cursor.fetchall()]
        
        if not user_liked_books:
            cursor.execute('''
                SELECT book_id, COUNT(*) as like_count
                FROM liked_books
                GROUP BY book_id
                HAVING COUNT(*) >= ?
                ORDER BY like_count DESC
                LIMIT ?
            ''', (min_likes, limit))
            return [row['book_id'] for row in cursor.fetchall()]
        
        # Find users who liked similar books
        placeholders = ','.join('?' * len(user_liked_books))
        cursor.execute(f'''
            SELECT DISTINCT lb.user_id, COUNT(*) as common_likes
            FROM liked_books lb
            WHERE lb.book_id IN ({placeholders})
            AND lb.user_id != ?
            GROUP BY lb.user_id
            ORDER BY common_likes DESC
            LIMIT ?
        ''', user_liked_books + [user_id, limit])
        
        similar_users = [row['user_id'] for row in cursor.fetchall()]
        
        if not similar_users:
            return []
        
        # Get books liked by similar users but not by current user
        user_placeholders = ','.join('?' * len(similar_users))
        # cursor.execute(f'''
        #     SELECT book_id, COUNT(*) as recommendation_score
        #     FROM liked_books
        #     WHERE user_id IN ({user_placeholders})
        #     AND book_id NOT IN ({placeholders})
        #     GROUP BY book_id
        #     ORDER BY recommendation_score DESC
        #     LIMIT ?
        # ''', similar_users + user_liked_books + [limit])
        cursor.execute(f'''
                SELECT book_id, COUNT(*) as recommendation_score
                FROM liked_books
                WHERE user_id IN ({user_placeholders})
                AND book_id NOT IN ({placeholders})
                GROUP BY book_id
                ORDER BY recommendation_score DESC
                LIMIT ?
            ''', similar_users + user_liked_books + [limit])
        
        return [row['book_id'] for row in cursor.fetchall()]
    
def get_similar_books(book_id, limit=12):
    """
    Get books that users who liked this book also liked.
    This is item-based collaborative filtering.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT l2.book_id, COUNT(*) as co_occurrence
            FROM liked_books l1
            JOIN liked_books l2 ON l1.user_id = l2.user_id
            WHERE l1.book_id = ?
              AND l2.book_id != ?
            GROUP BY l2.book_id
            ORDER BY co_occurrence DESC
            LIMIT ?
        ''', (book_id, book_id, limit))
        
        return [row['book_id'] for row in cursor.fetchall()]

# ============== USER STATISTICS ==============

def get_user_stats(user_id):
    """Get comprehensive user statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Liked books count
        cursor.execute('SELECT COUNT(*) as count FROM liked_books WHERE user_id = ?', (user_id,))
        liked_count = cursor.fetchone()['count']
        
        # Reading status counts
        cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM reading_progress 
            WHERE user_id = ? 
            GROUP BY status
        ''', (user_id,))
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM search_history 
            WHERE user_id = ? 
            AND created_at >= datetime('now', '-7 days')
        ''', (user_id,))
        recent_searches = cursor.fetchone()['count']
        
        return {
            'liked_count': liked_count,
            'want_to_read': status_counts.get('want_to_read', 0),
            'reading': status_counts.get('reading', 0),
            'completed': status_counts.get('completed', 0),
            'recent_searches': recent_searches,
            'total_books': liked_count + sum(status_counts.values())
        }

def get_like_count(book_id):
    """Get total number of likes for a book"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT COUNT(*) as count FROM liked_books WHERE book_id = ?',
            (book_id,)
        )
        return cursor.fetchone()['count']
    
def get_genre_stats(user_id, books_df):
    """
    Get genre distribution for user's library.
    Requires the books DataFrame to look up categories.
    Returns dict like: {'Fiction': 5, 'History': 3, ...}
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get all book IDs from all statuses + liked
        cursor.execute('''
            SELECT DISTINCT book_id FROM (
                SELECT book_id FROM reading_progress WHERE user_id = ?
                UNION
                SELECT book_id FROM liked_books WHERE user_id = ?
            )
        ''', (user_id, user_id))
        
        book_ids = [row['book_id'] for row in cursor.fetchall()]
        
        if not book_ids:
            return {}
        
        # Count genres from the DataFrame
        genre_counts = {}
        
        for book_id in book_ids:
            book = books_df[books_df['unique_values'] == book_id]
            if not book.empty:
                # Get categories (could be multiple separated by ;)
                categories = str(book.iloc[0].get('categories', ''))
                if categories and categories != 'nan':
                    # Split by semicolon or comma
                    genres = [g.strip() for g in categories.replace(';', ',').split(',')]
                    for genre in genres:
                        if genre:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        return genre_counts
##
def add_ratings_table():
    if not os.path.exists(DATABASE_NAME):
        print(f"❌ {DATABASE_NAME} not found.")
        print("Run setup_database.py first.")
        return
    
    print(f"🔄 Adding ratings table to {DATABASE_NAME}...")
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if ratings table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ratings'
        """)
        
        if cursor.fetchone():
            print("✅ Ratings table already exists!")
        else:
            # Create ratings table
            cursor.execute('''
                CREATE TABLE ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    book_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    UNIQUE(user_id, book_id)
                )
            ''')
            
            conn.commit()
            print("✅ Ratings table created successfully!")
        
        print("✅ Migration complete!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
    finally:
        conn.close()

# ============== RATINGS ==============

def add_rating(user_id, book_id, rating):
    """Add or update a book rating (1-5 stars)"""
    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''INSERT INTO ratings (user_id, book_id, rating) 
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id, book_id) 
                   DO UPDATE SET rating=?, updated_at=CURRENT_TIMESTAMP''',
                (user_id, book_id, rating, rating)
            )
            return True
        except Exception as e:
            print(f"Error adding rating: {e}")
            return False

def get_user_rating(user_id, book_id):
    """Get user's rating for a book"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT rating FROM ratings WHERE user_id = ? AND book_id = ?',
            (user_id, book_id)
        )
        result = cursor.fetchone()
        return result['rating'] if result else None

def get_average_rating(book_id):
    """Get average rating and count for a book"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT AVG(rating) as avg_rating, COUNT(*) as rating_count
               FROM ratings WHERE book_id = ?''',
            (book_id,)
        )
        result = cursor.fetchone()
        return {
            'average': round(result['avg_rating'], 1) if result['avg_rating'] else 0,
            'count': result['rating_count']
        }

def get_user_ratings(user_id, limit=None):
    """Get all ratings by a user"""
    with get_db() as conn:
        cursor = conn.cursor()
        query = '''SELECT book_id, rating, created_at 
                   FROM ratings 
                   WHERE user_id = ? 
                   ORDER BY created_at DESC'''
        if limit:
            query += f' LIMIT {limit}'
        cursor.execute(query, (user_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_highest_rated_books(limit=12, min_ratings=3):
    """Get books with highest average ratings
    
    Args:
        limit: Number of books to return
        min_ratings: Minimum number of ratings required for a book to be included
    
    Returns:
        List of dicts with book_id, avg_rating, and rating_count
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''SELECT book_id, 
                      AVG(rating) as avg_rating, 
                      COUNT(*) as rating_count
               FROM ratings 
               GROUP BY book_id
               HAVING COUNT(*) >= ?
               ORDER BY avg_rating DESC, rating_count DESC
               LIMIT ?''',
            (min_ratings, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
def fix_ratings():
    if not os.path.exists(DATABASE_NAME):
        print(f"❌ {DATABASE_NAME} not found.")
        print("Database doesn't exist yet. Run your app.py first to create it.")
        return
    
    print(f"🔄 Checking {DATABASE_NAME}...")
    
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if ratings table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ratings'
        """)
        
        if cursor.fetchone():
            print("✅ Ratings table already exists! No action needed.")
        else:
            print("📝 Creating ratings table...")
            
            # Create ratings table
            cursor.execute('''
                CREATE TABLE ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    book_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    UNIQUE(user_id, book_id)
                )
            ''')
            
            conn.commit()
            print("✅ Ratings table created successfully!")
        
        # Show all tables for verification
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n📊 Database tables: {', '.join(tables)}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_ratings()
    print("\n✅ Done! You can now use the rating system.")
    print("Restart your Flask app: python app.py")