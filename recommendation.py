import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class RecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.books_df = None
        self.similarity_cache = {}
        self.initialize_data()

    def initialize_data(self):
        """Initialize movie and book datasets"""
        movies = {
            "title": [
                "Interstellar",
                "Titanic",
                "Kal Ho Na Ho",
                "Wednesday",
                "Ra One",
                "Inception",
                "The Dark Knight",
                "Shutter Island",
                "Jawan",
                "3 Idiots",
                "Dangal",
                "Parasite",
                "Avengers: Endgame",
                "Gladiator",
                "Forrest Gump",
                "Zindagi Na Milegi Dobara",
                "The Wolf of Wall Street",
                "The Matrix",
                "Jab We Met",
                "Tumbbad",
                "Andhadhun",
                "Drishyam",
                "Black Swan",
                "The Revenant",
                "The Shawshank Redemption",
                "Kantara",
                "Super 30",
                "Oppenheimer",
                "Barfi!",
                "PK",
                "The Godfather",
                "Fight Club",
                "The Social Network",
                "K.G.F: Chapter 2",
                "Pathaan",
                "The Conjuring",
                "It",
                "Bhoot",
                "Bhool Bhulaiyaa",
                "Get Out",
                "Dhoom 2",
                "Mad Max: Fury Road",
                "Mission: Impossible - Fallout",
                "John Wick",
                "Logan",
                "Gully Boy",
                "Ludo",
                "The Prestige",
                "Eternal Sunshine of the Spotless Mind",
                "The Green Mile",
            ],
            "genre": [
                "Sci-Fi, Action",
                "Romance, Drama",
                "Romance, Drama",
                "Mystery, Teen",
                "Action, Sci-Fi",
                "Sci-Fi, Thriller",
                "Action, Crime",
                "Mystery, Thriller",
                "Action, Thriller",
                "Comedy, Drama",
                "Biography, Sports",
                "Thriller, Drama",
                "Action, Superhero",
                "Action, Drama",
                "Drama, Romance",
                "Adventure, Comedy",
                "Biography, Crime",
                "Sci-Fi, Action",
                "Romance, Comedy",
                "Horror, Fantasy",
                "Thriller, Mystery",
                "Thriller, Crime",
                "Psychological, Thriller",
                "Adventure, Drama",
                "Drama, Crime",
                "Fantasy, Thriller",
                "Biography, Drama",
                "Biography, Thriller",
                "Romance, Comedy",
                "Comedy, Drama",
                "Crime, Drama",
                "Drama, Thriller",
                "Biography, Drama",
                "Action, Drama",
                "Action, Spy Thriller",
                "Horror, Thriller",
                "Horror, Mystery",
                "Horror, Thriller",
                "Horror, Comedy",
                "Horror, Mystery",
                "Action, Thriller",
                "Action, Adventure",
                "Action, Thriller",
                "Action, Thriller",
                "Action, Drama",
                "Musical, Drama",
                "Comedy, Crime",
                "Mystery, Drama",
                "Romance, Drama",
                "Drama, Fantasy",
            ],
        }

        books = {
            "title": [
                "The Old Man and the Sea",
                "Doctor Zhivago",
                "The God of Small Things",
                "Snow",
                "The Golden Notebook",
                "Harry Potter and the Sorcerer’s Stone",
                "1984",
                "Pride and Prejudice",
                "To Kill a Mockingbird",
                "The Great Gatsby",
                "The Lord of the Rings",
                "Moby-Dick",
                "Crime and Punishment",
                "Don Quixote",
                "The Catcher in the Rye",
                "One Hundred Years of Solitude",
                "Brave New World",
                "The Alchemist",
                "The Picture of Dorian Gray",
                "The Hunger Games",
                "A Tale of Two Cities",
                "Jane Eyre",
                "Wuthering Heights",
                "The Book Thief",
                "Les Misérables",
                "The Road",
                "The Kite Runner",
                "The Bell Jar",
                "Dracula",
                "Frankenstein",
                "The Hobbit",
                "Life of Pi",
                "War and Peace",
                "The Night Circus",
                "Dune",
                "The Shadow of the Wind",
                "The Name of the Wind",
                "The Stand",
                "The Handmaid’s Tale",
                "The Girl with the Dragon Tattoo",
                "The Fault in Our Stars",
                "Shantaram",
                "The Midnight Library",
                "A Man Called Ove",
                "Percy Jackson & The Lightning Thief",
                "The Subtle Art of Not Giving a F*ck",
                "Sapiens",
                "Atomic Habits",
                "Rich Dad Poor Dad",
                "Ikigai",
                "The Power of Now",
            ],
            "genre": [
                "Adventure",
                "Romance, Fiction",
                "Fiction",
                "Drama, Fiction",
                "Fiction",
                "Fantasy",
                "Dystopian, Political",
                "Romance, Fiction",
                "Drama, Fiction",
                "Tragedy, Fiction",
                "Fantasy, Adventure",
                "Adventure, Fiction",
                "Psychological, Fiction",
                "Satire, Adventure",
                "Coming-of-Age, Fiction",
                "Magical Realism, Fiction",
                "Dystopian, Sci-Fi",
                "Philosophical, Fiction",
                "Gothic, Fiction",
                "Dystopian, Sci-Fi",
                "Historical, Fiction",
                "Romance, Fiction",
                "Gothic, Romance",
                "Historical, Drama",
                "Historical, Fiction",
                "Post-Apocalyptic, Fiction",
                "Drama, Fiction",
                "Psychological, Fiction",
                "Horror, Fiction",
                "Horror, Fiction",
                "Fantasy, Adventure",
                "Adventure, Fiction",
                "Historical, Fiction",
                "Fantasy, Fiction",
                "Sci-Fi, Fantasy",
                "Mystery, Fiction",
                "Fantasy, Fiction",
                "Horror, Fiction",
                "Dystopian, Fiction",
                "Mystery, Thriller",
                "Romance, Drama",
                "Adventure, Fiction",
                "Contemporary, Fiction",
                "Contemporary, Fiction",
                "Fantasy, Adventure",
                "Self-Help, Non-Fiction",
                "History, Non-Fiction",
                "Self-Help, Non-Fiction",
                "Finance, Non-Fiction",
                "Self-Help, Non-Fiction",
                "Spirituality, Non-Fiction",
            ],
        }

        self.movies_df = pd.DataFrame(movies)
        self.books_df = pd.DataFrame(books)

    def preprocess_genres(self, genre_string):
        """Normalize and clean genre strings"""
        return [g.strip().lower() for g in genre_string.split(",")]

    def get_similarity_matrix(self, df):
        """Calculate or retrieve cached similarity matrix"""
        cache_key = hash(tuple(df["genre"]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(df["genre"])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        self.similarity_cache[cache_key] = cosine_sim
        return cosine_sim

    def recommend_items(self, df, item_title, top_n=3):
        """Recommend items based on content similarity"""
        try:
            indexes = pd.Series(df.index, index=df["title"]).drop_duplicates()
            if item_title not in indexes:
                print(f"'{item_title}' not found in database.")
                return []

            ind = indexes[item_title]
            cosine_sim = self.get_similarity_matrix(df)
            sim_scores = list(enumerate(cosine_sim[ind]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
                1 : top_n + 1
            ]
            item_indices = [i[0] for i in sim_scores]

            recommendations = df["title"].iloc[item_indices].tolist()
            scores = [score[1] for score in sim_scores]

            return list(zip(recommendations, scores))

        except Exception as e:
            print(f"Error in recommendation: {e}")
            return []

    def collaborative_filtering(self, user_preferences, items_df, top_n=3):
        """Collaborative filtering using shared preferences"""
        try:
            user_recommendations = defaultdict(float)
            for pref in user_preferences:
                similar_items = self.recommend_items(items_df, pref, top_n)
                for item, score in similar_items:
                    user_recommendations[item] += score

            return sorted(
                user_recommendations.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
        except Exception as e:
            print(f"Error in collaborative filtering: {e}")
            return []

    def display_recommendations(self, recommendations):
        """Display formatted recommendations"""
        if not recommendations:
            print("No recommendations found.")
            return

        print("\nRecommended items:")
        for i, (item, score) in enumerate(recommendations, 1):
            print(f"{i}. {item} (Similarity: {score:.2f})")

    def get_valid_input(self, prompt, valid_items):
        """Get and validate user input"""
        while True:
            item = input(prompt).strip()
            if item in valid_items:
                return item
            print(f"Invalid input. Please choose from available items.")


def main():
    recommender = RecommendationSystem()

    while True:
        print("\n=== Recommendation System ===")
        print("1. Get Movie Recommendations")
        print("2. Get Book Recommendations")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == "3":
            print("Thank you for using the recommendation system!")
            break

        elif choice in ["1", "2"]:
            # Display available items
            df = recommender.movies_df if choice == "1" else recommender.books_df
            item_type = "movie" if choice == "1" else "book"

            print(f"\nAvailable {item_type}s:")
            for i, title in enumerate(df["title"], 1):
                print(f"{i}. {title}")

            # Get user preferences
            num_prefs = int(
                input(
                    f"\nHow many {item_type}s would you like to base your recommendations on? "
                )
            )
            user_preferences = []

            for i in range(num_prefs):
                prompt = f"\nEnter {item_type} {i+1}: "
                item = recommender.get_valid_input(prompt, set(df["title"]))
                user_preferences.append(item)

            # Get and display recommendations
            recommendations = recommender.collaborative_filtering(
                user_preferences, df, top_n=3
            )
            recommender.display_recommendations(recommendations)

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
