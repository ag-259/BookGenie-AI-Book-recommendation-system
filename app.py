from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import database as db
from openai import OpenAI
import os

load_dotenv()   
app = Flask(__name__)
app.secret_key = 'ayush12092005'  
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

books = pd.read_csv("cleaned_file.csv")

books["large_thumbnail"] = books["image"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "https://via.placeholder.com/200x280?text=No+Cover",
    books["large_thumbnail"],
)


db_books = Chroma(
    persist_directory="chroma_store",
    embedding_function=OpenAIEmbeddings()
)


categories = ["All"] + sorted(
    books["simple_categories"].astype(str).unique().tolist()
)

tones = [
    "All",
    "Happy",
    "Surprising",
    "Angry",
    "Suspenseful",
    "Sad",
]


def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 150,
    final_top_k: int = 104
):
    # 1. Semantic search
    recs = db_books.similarity_search(query, k=initial_top_k)

    if not recs:
        return pd.DataFrame()

    # 2. Extract unique_values from page_content (SAME AS GRADIO)
    book_ids = []
    for rec in recs:
        try:
            book_ids.append(
                int(rec.page_content.strip('"').split()[0])
            )
        except:
            continue

    if not book_ids:
        return pd.DataFrame()

    # 3. Filter dataset
    book_recs = books[books["unique_values"].isin(book_ids)]

    # 4. Tone sorting
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    # 5. Category filter
    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ]

    return book_recs.head(final_top_k)



def format_book_results(df: pd.DataFrame):
    results = []

    for _, row in df.iterrows():
        authors = str(row.get("authors", ""))
        authors_split = authors.split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = (
                ", ".join(authors_split[:-1]) + f", and {authors_split[-1]}"
            )
        else:
            authors_str = authors

        results.append(
            {
                "unique_values": int(row["unique_values"]),
                "title": row.get("Title", ""),
                "authors": authors_str,
                "thumbnail": row.get("large_thumbnail", ""),
                "category": row.get("simple_categories", ""),
                "previewLink": row.get("previewLink", ""),
            }
        )

    return results

tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
book = joblib.load("books_df.pkl")

book.fillna("", inplace=True)

GENRES = [
    "All", "Fantasy", "Romance", "Mystery",
    "Horror", "Science Fiction", "Juvenile"
]

# ---------------- HYBRID SEARCH ----------------
def keyword_recomendation(query, genre=None, top_k=12):
    if not query.strip():
        return []

    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    books_copy = book.copy()
    books_copy["similarity"] = scores

    if genre and genre != "All":
        books_copy = books_copy[
            books_copy["categories"].str.contains(genre, case=False, na=False)
        ]

    books_copy = books_copy.sort_values(
        by="similarity", ascending=False
    )

    return books_copy.head(top_k).to_dict(orient="records")


# ---------------- GET BOOKS BY CATEGORIES ----------------
def get_books_by_categories():
    """Get random books organized by specific categories"""
    
    categories_map = {
        "Fiction": "Fiction",
        "Juvenile Fiction": "Juvenile Fiction",
        "History": "History",
        "Biography & Autobiography": "Biography & Autobiography",
        "Comics & Graphic Novels": "Comics & Graphic Novels",
        "Poetry": "Poetry",
        "Nature": "Nature",
        "Travel": "Travel"
    }
    
    books_by_category = {}
    
    for category_key, category_search in categories_map.items():
        # Filter books that contain this genre in their categories
        category_books = books[
            books["categories"].str.contains(category_search, case=False, na=False)
        ]
        
        if len(category_books) > 0:
            # Get random 30 books from this category
            sample_size = min(30, len(category_books))
            sampled = category_books.sample(n=sample_size)
            books_by_category[category_key] = sampled.to_dict(orient="records")
        else:
            books_by_category[category_key] = []
    
    return books_by_category

def get_personalized_recommendations(user_id, genre=None, top_n=12):
    liked_books = db.get_liked_books(user_id)

    if not liked_books:
        return []

    # Take the most recently liked book
    latest_book_id = liked_books[-1]

    latest_book = books[books["unique_values"] == latest_book_id]

    if latest_book.empty:
        return []

    # Use ONLY the latest liked book title (optionally add categories)
    title = latest_book.iloc[0]["Title"]
    categories = latest_book.iloc[0].get("categories", "")

    query = f"{title} {categories}".strip()

    return keyword_recomendation(
        query=query,    
        genre=genre,
        top_k=top_n
    )
    

def get_current_user_id():
    """Get current logged-in user ID"""
    return session.get('user_id')

def require_login():
    """Decorator function to require login"""
    user_id = get_current_user_id()
    if not user_id:
        flash('Please log in to access this page', 'error')
        return redirect(url_for('login_page'))
    return None

def get_books_from_ids(book_ids):
    """Get book details from list of IDs"""
    if not book_ids:
        return []
    books_df = books[books['unique_values'].isin(book_ids)]
    return books_df.to_dict(orient='records')

#Authentication routes
@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for('register_page'))
        
        success, result = db.create_user(username, email, password)
        
        if success:
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login_page'))
        else:
            flash(result, "error")
            return redirect(url_for('register_page'))
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = request.form.get("remember")
        
        success, user_data = db.login_user(email, password)
        
        if success:
            session['user_id'] = user_data['id']
            session['username'] = user_data['username']
            session['email'] = user_data['email']
            session['theme'] = user_data.get('theme', 'dark')
            
            if remember:
                session.permanent = True
            
            flash("Login successful!", "success")
            return redirect(url_for('search'))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for('login_page'))
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out", "success")
    return redirect(url_for('login_page'))

# ---------------- SEARCH PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def search():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    user_id = get_current_user_id()
    results = []
    selected_genre = "All"
    query = ""

    if request.method == "POST":
        query = request.form.get("query")
        selected_genre = request.form.get("genre")
        results = keyword_recomendation(query, selected_genre)

    # Get books organized by categories
    category_books = get_books_by_categories()

     # Get personalized recommendations
    personalized = get_personalized_recommendations(user_id, genre=None, top_n=12)
    
    # Get collaborative recommendations
    collaborative_ids = db.get_collaborative_recommendations(user_id, limit=12,min_likes=2)
    collaborative = get_books_from_ids(collaborative_ids)
    if not collaborative_ids:
        collaborative_ids = db.get_collaborative_recommendations(user_id, limit=12, min_likes=2  )
    # Get highest rated books
    highest_rated_data = db.get_highest_rated_books(limit=12, min_ratings=3)
    highest_rated_ids = [item['book_id'] for item in highest_rated_data]
    highest_rated = get_books_from_ids(highest_rated_ids)
    # Get category books
    category_books = get_books_by_categories()

    return render_template(
        "home2.html",
        books=results,
        genres=GENRES,
        selected_genre=selected_genre,
        category_books=category_books,
        personalized=personalized,
        collaborative=collaborative,
        highest_rated=highest_rated,
        search_query=query
    )

# ------Testing -------------
def get_similar_books_content_based(book_id, top_k=12):
    # Find the book in the dataset
    book_row = books[books["unique_values"] == book_id]
    
    if book_row.empty:
        return []
    
    book = book_row.iloc[0]
    query_parts = []
    
    if pd.notna(book.get("Title")):
        query_parts.append(str(book["Title"]))
    
    if pd.notna(book.get("categories")):
        query_parts.append(str(book["categories"]))
    
    query = " ".join(query_parts)
    
    if not query.strip():
        return []
    
    # Use keyword_recomendation to find similar books
    results = keyword_recomendation(query, genre=None, top_k=top_k + 1)
    
    # Filter out the book itself
    similar_books = [
        b for b in results 
        if b.get("unique_values") != book_id
    ]
    
    return similar_books[:top_k]


def get_similar_books_semantic(book_id, top_k=12):
    # Find the book
    book_row = books[books["unique_values"] == book_id]
    
    if book_row.empty:
        return []
    
    book = book_row.iloc[0]
    
    # Build query from title + description
    query_parts = []
    
    if pd.notna(book.get("Title")):
        query_parts.append(str(book["Title"]))
    
    if pd.notna(book.get("description")):
        # Take first 200 chars of description to avoid overwhelming the query
        desc = str(book["description"])[:200]
        query_parts.append(desc)
    
    query = " ".join(query_parts)
    
    if not query.strip():
        return []
    
    # Use semantic search
    try:
        recs_df = retrieve_semantic_recommendations(
            query=query,
            category="All",
            tone="All",
            initial_top_k=50,
            final_top_k=top_k + 1
        )
        
        # Convert to dict and filter out the current book
        results = recs_df.to_dict(orient="records")
        similar_books = [
            b for b in results 
            if b.get("unique_values") != book_id
        ]
        
        return similar_books[:top_k]
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return []
    
def get_similar_books_hybrid(book_id, top_k=12):
    # Get results from both methods
    tfidf_results = get_similar_books_content_based(book_id, top_k=8)
    semantic_results = get_similar_books_semantic(book_id, top_k=8)
    
    # Merge and deduplicate
    seen_ids = set()
    combined = []
    
    # Interleave results (alternate between sources for diversity)
    max_len = max(len(tfidf_results), len(semantic_results))
    
    for i in range(max_len):
        if i < len(tfidf_results):
            book = tfidf_results[i]
            book_id_val = book.get("unique_values")
            if book_id_val not in seen_ids:
                combined.append(book)
                seen_ids.add(book_id_val)
        
        if i < len(semantic_results):
            book = semantic_results[i]
            book_id_val = book.get("unique_values")
            if book_id_val not in seen_ids:
                combined.append(book)
                seen_ids.add(book_id_val)
        
        if len(combined) >= top_k:
            break
    
    return combined[:top_k]

# RAG retrieval function
def retrieve_books_for_rag(query: str, top_k: int = 10):
    """
    Retrieve relevant books for RAG context.
    Uses semantic search to find books matching the query.
    """
    try:
        # Use existing semantic search
        recs_df = retrieve_semantic_recommendations(
            query=query,
            category="All",
            tone="All",
            initial_top_k=50,
            final_top_k=top_k
        )
        
        if recs_df.empty:
            return []
        
        # Format books for context
        books_context = []
        for _, book in recs_df.iterrows():
            books_context.append({
                'id': int(book['unique_values']),
                'title': book.get('Title', ''),
                'authors': book.get('authors', ''),
                'description': book.get('description', '')[:300],  # Truncate long descriptions
                'categories': book.get('categories', ''),
                'rating': book.get('averageRating', 0),
                'emotions': {
                    'joy': float(book.get('joy', 0)),
                    'surprise': float(book.get('surprise', 0)),
                    'sadness': float(book.get('sadness', 0)),
                    'fear': float(book.get('fear', 0))
                }
            })
        
        return books_context
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []


def generate_rag_response(user_query: str, books_context: list, user_id=None):
    """
    Generate AI response using retrieved books as context.
    """
    
    # Build context string from retrieved books
    context_parts = []
    for i, book in enumerate(books_context, 1):
        emotions_str = ", ".join([
            f"{k}: {v:.2f}" for k, v in book['emotions'].items() if v > 0.1
        ])
        
        context_parts.append(
            f"Book {i}:\n"
            f"Title: {book['title']}\n"
            f"Author: {book['authors']}\n"
            f"Categories: {book['categories']}\n"
            f"Description: {book['description']}\n"
            f"Emotional Tone: {emotions_str}\n"
            f"ID: {book['id']}"
        )
    
    context_text = "\n\n".join(context_parts)
    
    # Get user's reading history for personalization (optional)
    user_context = ""
    if user_id:
        try:
            liked_books = db.get_liked_books(user_id)
            if liked_books:
                user_context = f"\n\nUser has previously liked {len(liked_books)} books."
        except:
            pass
    
    # System prompt
    system_prompt = """You are BookGenie, an expert book recommendation assistant. 

                        Your job is to help users discover books based on their questions and preferences.

                        When recommending books:
                        1. Use ONLY books from the provided context
                        2. Be conversational and enthusiastic
                        3. Explain WHY each book matches their request
                        4. Mention relevant emotional tones when appropriate
                        5. Keep recommendations to 3-5 books unless asked for more
                        6. Include the book ID in format [Book ID: X] after each title

                        If the user asks something unrelated to books, politely redirect them to book-related topics."""

    user_prompt = f"""Based on these books from our database:

{context_text}
{user_context}

User question: {user_query}

Please recommend the most suitable books and explain why they match the request."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Sorry, I'm having trouble generating a response right now. Please try again."


def extract_book_ids_from_response(response_text: str):
    """
    Extract book IDs from RAG response to show book cards.
    Looks for pattern [Book ID: X]
    """
    import re
    pattern = r'\[Book ID:\s*(\d+)\]'
    matches = re.findall(pattern, response_text)
    return [int(book_id) for book_id in matches]


# -------ROUTES --------
@app.route("/genie")
def index():
    return render_template(
        "index2.html",
        categories=categories,
        tones=tones,
    )


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    query = request.args.get("query", "").strip()
    category = request.args.get("category", "All")
    tone = request.args.get("tone", "All")

    if not query:
        return jsonify([])

    try:
        recs = retrieve_semantic_recommendations(
            query=query,
            category=category,
            tone=tone,
        )

        results = format_book_results(recs)
        return jsonify(results)

    except Exception as e:
        print("API ERROR:", e)
        return jsonify({"error": str(e)}), 500


# @app.route("/book/<int:unique_values>")
# def book_details(unique_values):
#     book_df = books[books["unique_values"] == unique_values]

#     if book_df.empty:
#         return "Book not found", 404

#     book = book_df.iloc[0].to_dict()
#     return render_template("book2.html", book=book)

@app.route("/book/<int:unique_values>")
def book_details(unique_values):
    book_df = books[books["unique_values"] == unique_values]

    if book_df.empty:
        return "Book not found", 404

    book = book_df.iloc[0].to_dict()
    
    # Get user-specific data
    is_liked = False
    current_status = None
    like_count = 0
    user_id = get_current_user_id()
    rating_info = {'average': 0, 'count': 0}
    
    if user_id:
        is_liked = db.is_book_liked(user_id, unique_values)
        try:
            current_status = db.get_reading_status(user_id, unique_values)
        except:
            current_status = None

        user_rating = db.get_user_rating(user_id, unique_values)

    like_count = db.get_like_count(unique_values)
    similar_books = get_similar_books_hybrid(unique_values, top_k=12)
    
    return render_template(
        "book2.html",
        book=book,
        is_liked=is_liked,
        current_status=current_status,
        similar_books=similar_books,
        like_count=like_count,
        user_rating=user_rating,
        rating_info=rating_info
    )

# Reading status and liked books routes
@app.route("/set_status/<int:unique_values>/<status>", methods=["POST"])
def set_status(unique_values, status):
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    
    if status in ['want_to_read', 'reading', 'completed']:
        db.set_reading_status(user_id, unique_values, status)
        flash(f"Book marked as {status.replace('_', ' ').title()}", "success")
    
    return redirect(url_for('book_details', unique_values=unique_values))

@app.route("/readlist/<status>")
def readlist(status):
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    book_ids = db.get_books_by_status(user_id, status)
    books_list = get_books_from_ids(book_ids)
    
    status_titles = {
        'want_to_read': 'Want to Read',
        'reading': 'Currently Reading',
        'completed': 'Completed'
    }
    
    return render_template(
        "readlist.html",
        books=books_list,
        status=status,
        title=status_titles.get(status, status)
    )

@app.route("/liked")
def liked_page():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    liked_ids = db.get_liked_books(user_id)
    liked_books = get_books_from_ids(liked_ids)
    
    return render_template("liked.html", liked_books=liked_books)

@app.route("/api/toggle_like/<int:unique_values>", methods=["POST"])
def api_toggle_like(unique_values):
    """API endpoint for AJAX like toggling with count"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        is_liked = db.is_book_liked(user_id, unique_values)
        
        if is_liked:
            db.remove_liked_book(user_id, unique_values)
            liked = False
        else:
            db.add_liked_book(user_id, unique_values)
            liked = True
        
        # Get updated count
        like_count = db.get_like_count(unique_values)
        
        return jsonify({
            "success": True,
            "liked": liked,
            "book_id": unique_values,
            "like_count": like_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Profile routes
@app.route("/profile")
def profile():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    stats = db.get_user_stats(user_id)

    genre_stats = db.get_genre_stats(user_id, books)
    
    # Get top 5 genres for the pie chart
    top_genres = dict(sorted(genre_stats.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # If user has more than 5 genres, group the rest as "Other"
    if len(genre_stats) > 5:
        other_count = sum(list(genre_stats.values())[5:])
        top_genres['Other'] = other_count
    
    return render_template(
        "profile.html",
        stats=stats,
        genre_stats=top_genres
    )

@app.route("/update_theme", methods=["POST"])
def update_theme():
    user_id = get_current_user_id()
    if user_id:
        theme = request.form.get('theme', 'dark')
        db.update_user_theme(user_id, theme)
        session['theme'] = theme
        return jsonify({"success": True})
    return jsonify({"success": False}), 401

@app.route("/want-to-read")
def want_to_read():
    """Books user wants to read"""
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    book_ids = db.get_books_by_status(user_id, 'want_to_read')
    books_list = get_books_from_ids(book_ids)
    
    return render_template(
        "reading_status.html",
        books=books_list,
        status='want_to_read',
        title='Want to Read',
        icon='📌',
        empty_message="No books in your 'Want to Read' list yet. Start adding books you're interested in!"
    )


@app.route("/reading")
def currently_reading():
    """Books user is currently reading"""
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    book_ids = db.get_books_by_status(user_id, 'reading')
    books_list = get_books_from_ids(book_ids)
    
    return render_template(
        "reading_status.html",
        books=books_list,
        status='reading',
        title='Currently Reading',
        icon='📖',
        empty_message="You're not reading any books right now. Pick one from your 'Want to Read' list!"
    )


@app.route("/completed")
def completed():
    """Books user has completed"""
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    book_ids = db.get_books_by_status(user_id, 'completed')
    books_list = get_books_from_ids(book_ids)
    
    return render_template(
        "reading_status.html",
        books=books_list,
        status='completed',
        title='Completed',
        icon='✅',
        empty_message="You haven't finished any books yet. Keep reading!"
    )


@app.route("/library")
def library():
    """All books across all statuses (Want to Read + Reading + Completed)"""
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    user_id = get_current_user_id()
    
    # Get books from all three statuses
    want_to_read_ids = db.get_books_by_status(user_id, 'want_to_read')
    reading_ids = db.get_books_by_status(user_id, 'reading')
    completed_ids = db.get_books_by_status(user_id, 'completed')
    
    # Combine and deduplicate (in case a book somehow has multiple statuses)
    all_ids = list(set(want_to_read_ids + reading_ids + completed_ids))
    
    books_list = get_books_from_ids(all_ids)
    
    # Add status info to each book so we can show badges
    for book in books_list:
        book_id = book['unique_values']
        if book_id in want_to_read_ids:
            book['current_status'] = 'want_to_read'
        elif book_id in reading_ids:
            book['current_status'] = 'reading'
        elif book_id in completed_ids:
            book['current_status'] = 'completed'
        else:
            book['current_status'] = None
    
    # Stats for the header
    stats = {
        'want_to_read': len(want_to_read_ids),
        'reading': len(reading_ids),
        'completed': len(completed_ids),
        'total': len(all_ids)
    }
    
    return render_template(
        "library.html",
        books=books_list,
        stats=stats,
        empty_message="Your library is empty. Start adding books by clicking the reading status buttons on any book page!"
    )

@app.route("/api/rate_book/<int:unique_values>", methods=["POST"])
def api_rate_book(unique_values):
    """API endpoint for book rating"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        rating = data.get('rating')
        
        if not rating or rating < 1 or rating > 5:
            return jsonify({"error": "Invalid rating"}), 400
        
        # Save rating
        db.add_rating(user_id, unique_values, rating)
        
        # Get updated average
        rating_info = db.get_average_rating(unique_values)
        
        return jsonify({
            "success": True,
            "rating": rating,
            "rating_info": rating_info
        })
    except Exception as e:
        print(f"Rating error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask")
def ask_page():
    """RAG interface page"""
    redirect_response = require_login()
    if redirect_response:
        return redirect_response
    
    return render_template("ask.html")

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """RAG API endpoint"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        
        # Step 1: Retrieve relevant books
        books_context = retrieve_books_for_rag(query, top_k=10)
        
        if not books_context:
            return jsonify({
                "response": "I couldn't find any books matching your query. Try rephrasing or searching for something else.",
                "books": []
            })
        
        # Step 2: Generate AI response
        ai_response = generate_rag_response(query, books_context, user_id)
        
        # Step 3: Extract recommended book IDs
        recommended_ids = extract_book_ids_from_response(ai_response)
        
        # Step 4: Get full book details
        recommended_books = []
        if recommended_ids:
            recommended_books = get_books_from_ids(recommended_ids)
        else:
            # Fallback: show top 5 from retrieved books
            recommended_books = get_books_from_ids([b['id'] for b in books_context[:5]])
        
        return jsonify({
            "response": ai_response,
            "books": recommended_books,
            "query": query
        })
    
    except Exception as e:
        print(f"Ask API error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)



