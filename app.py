import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Meal Planner",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🍽️ AI Meal Planner")

st.write(
    "Create a personalized meal plan based on your age and diet, "
    "and get recipes from your recipe book using AI."
)


# =========================================================
# USER INFORMATION
# =========================================================

st.header("👤 Personal Information")

name = st.text_input(
    "Enter your name"
)

age = st.number_input(
    "Enter your age",
    min_value=5,
    max_value=100,
    value=18,
    step=1
)

diet = st.selectbox(
    "Select your diet",
    [
        "Vegetarian",
        "Non-Vegetarian",
        "Vegan",
        "Eggetarian"
    ]
)


# =========================================================
# AGE GROUP
# =========================================================

if age <= 12:
    age_group = "Child"

elif age <= 17:
    age_group = "Teenager"

elif age <= 59:
    age_group = "Adult"

else:
    age_group = "Senior"


# =========================================================
# MEAL TIMINGS BASED ON AGE
# =========================================================

meal_times = {

    "Child": {
        "Breakfast": "7:30 AM",
        "Mid-Morning Snack": "10:30 AM",
        "Lunch": "1:00 PM",
        "Evening Snack": "4:30 PM",
        "Dinner": "7:30 PM"
    },

    "Teenager": {
        "Breakfast": "7:00 AM",
        "Mid-Morning Snack": "10:30 AM",
        "Lunch": "1:30 PM",
        "Evening Snack": "5:00 PM",
        "Dinner": "8:00 PM"
    },

    "Adult": {
        "Breakfast": "7:30 AM",
        "Mid-Morning Snack": "11:00 AM",
        "Lunch": "1:00 PM",
        "Evening Snack": "5:00 PM",
        "Dinner": "8:00 PM"
    },

    "Senior": {
        "Breakfast": "7:30 AM",
        "Mid-Morning Snack": "10:30 AM",
        "Lunch": "12:30 PM",
        "Evening Snack": "4:30 PM",
        "Dinner": "7:00 PM"
    }
}


# =========================================================
# FOOD SUGGESTIONS
# =========================================================

food_suggestions = {

    "Vegetarian": {

        "Breakfast":
            "Idli, dosa, vegetable upma or poha",

        "Lunch":
            "Rice, dal, vegetables and curd",

        "Snack":
            "Fruits, nuts or sprouts",

        "Dinner":
            "Chapati, vegetable curry and dal"
    },

    "Non-Vegetarian": {

        "Breakfast":
            "Eggs, dosa, idli or vegetable upma",

        "Lunch":
            "Rice, vegetables and chicken or fish",

        "Snack":
            "Fruits, nuts or boiled eggs",

        "Dinner":
            "Chapati with chicken, fish or vegetables"
    },

    "Vegan": {

        "Breakfast":
            "Oats, poha, idli or vegetable dosa",

        "Lunch":
            "Rice, dal, vegetables and salad",

        "Snack":
            "Fruits, nuts or roasted chickpeas",

        "Dinner":
            "Chapati, vegetable curry and dal"
    },

    "Eggetarian": {

        "Breakfast":
            "Eggs, dosa, idli or vegetable upma",

        "Lunch":
            "Rice, dal, vegetables and eggs",

        "Snack":
            "Fruits, nuts or boiled eggs",

        "Dinner":
            "Chapati, vegetables and egg curry"
    }
}


# =========================================================
# GENERATE MEAL PLAN
# =========================================================

if st.button("🍽️ Generate Meal Plan"):

    if name.strip() == "":

        st.warning(
            "Please enter your name."
        )

    else:

        st.success(
            f"Hello {name}! Your personalized meal plan is ready."
        )

        st.info(
            f"Age: {age} | "
            f"Age Group: {age_group} | "
            f"Diet: {diet}"
        )

        st.header("📅 Your Daily Meal Plan")

        times = meal_times[age_group]

        foods = food_suggestions[diet]

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # LEFT COLUMN
        # -------------------------------------------------

        with col1:

            st.subheader("🌅 Breakfast")

            st.write(
                "⏰ " + times["Breakfast"]
            )

            st.write(
                foods["Breakfast"]
            )

            st.subheader("🍎 Mid-Morning Snack")

            st.write(
                "⏰ " + times["Mid-Morning Snack"]
            )

            st.write(
                foods["Snack"]
            )

            st.subheader("🍛 Lunch")

            st.write(
                "⏰ " + times["Lunch"]
            )

            st.write(
                foods["Lunch"]
            )

        # -------------------------------------------------
        # RIGHT COLUMN
        # -------------------------------------------------

        with col2:

            st.subheader("☕ Evening Snack")

            st.write(
                "⏰ " + times["Evening Snack"]
            )

            st.write(
                foods["Snack"]
            )

            st.subheader("🌙 Dinner")

            st.write(
                "⏰ " + times["Dinner"]
            )

            st.write(
                foods["Dinner"]
            )

        st.success(
            "Meal plan generated successfully! ✅"
        )


# =========================================================
# RECIPE BOOK AI
# =========================================================

st.divider()

st.header("📚🤖 Recipe Book AI")

st.write(
    "Ask for any recipe. The AI first checks your recipe book. "
    "If the recipe is not available, AI will create a new recipe."
)


# =========================================================
# LOAD RAG DATABASE
# =========================================================

@st.cache_resource
def load_recipe_database():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    database = Chroma(
        persist_directory="recipe_database",
        embedding_function=embeddings
    )

    return database


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_ai_model():

    model = OllamaLLM(
        model="gemma3:1b"
    )

    return model


# =========================================================
# LOAD AI AND DATABASE
# =========================================================

try:

    database = load_recipe_database()

    ai_model = load_ai_model()

    st.success(
        "🤖 Recipe AI is ready!"
    )


    # =====================================================
    # RECIPE QUESTION
    # =====================================================

    question = st.text_input(
        "💬 What recipe do you want?"
    )


    # =====================================================
    # GENERATE RECIPE BUTTON
    # =====================================================

    if st.button("🍳 Ask Recipe AI"):

        if question.strip() == "":

            st.warning(
                "Please enter a recipe name or question."
            )

        else:

            # =============================================
            # SEARCH RECIPE BOOK
            # =============================================

            with st.spinner(
                "📚 Searching your recipe book..."
            ):

                results = database.similarity_search(
                    question,
                    k=4
                )


                # =========================================
                # GET BOOK CONTENT
                # =========================================

                context = "\n\n".join(
                    document.page_content
                    for document in results
                )


                # =========================================
                # CHECK IF RECIPE EXISTS
                # =========================================

                check_prompt = f"""

You are checking a recipe book.

RECIPE BOOK CONTENT:
{context}

USER REQUEST:
{question}

Determine whether the requested recipe or the
information needed to answer the request is
actually present in the recipe book.

Reply with ONLY one of these:

FOUND

or

NOT_FOUND

"""


                check_result = ai_model.invoke(
                    check_prompt
                ).strip().upper()


            # =================================================
            # RECIPE FOUND
            # =================================================

            if "FOUND" in check_result:

                st.success(
                    "📚 Recipe found in your recipe book!"
                )

                with st.spinner(
                    "📖 Preparing the recipe from your book..."
                ):

                    recipe_prompt = f"""

You are a helpful recipe assistant.

Use ONLY the recipe book information below.

RECIPE BOOK:
{context}

USER REQUEST:
{question}

Provide the recipe in a simple format.

Include:

🍽️ Recipe Name

🥕 Ingredients

👨‍🍳 Preparation Steps

🔥 Cooking Instructions

⏱️ Cooking Time

🍴 Serving Size

Do not invent information that is not supported
by the recipe book.

"""


                    answer = ai_model.invoke(
                        recipe_prompt
                    )


                st.subheader(
                    "📖 Recipe From Your Book"
                )

                st.write(
                    answer
                )


            # =================================================
            # RECIPE NOT FOUND
            # =================================================

            else:

                st.info(
                    "📚 This recipe was not found in your book."
                )

                st.write(
                    "🤖 AI will create a new recipe for you."
                )

                with st.spinner(
                    "🤖 Creating your recipe..."
                ):

                    generate_prompt = f"""

You are an expert recipe assistant.

The user requested:

{question}

The requested recipe is NOT available in
the provided recipe book.

Create a practical and easy-to-follow recipe.

Use this format:

🍽️ RECIPE NAME

🥕 INGREDIENTS
- Ingredient 1
- Ingredient 2
- Ingredient 3

⏱️ PREPARATION TIME
Give an approximate time.

🔥 COOKING TIME
Give an approximate time.

👨‍🍳 PREPARATION STEPS

1. Step one
2. Step two
3. Step three
4. Step four

🍴 SERVING
Give the approximate number of servings.

💡 HEALTHY TIP
Give one simple healthy suggestion.

Make the recipe clear, practical and easy to
prepare at home.

"""


                    generated_recipe = ai_model.invoke(
                        generate_prompt
                    )


                st.success(
                    "🤖 New recipe generated by AI!"
                )

                st.subheader(
                    "🍳 AI Generated Recipe"
                )

                st.write(
                    generated_recipe
                )


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as e:

    st.error(
        "❌ Recipe AI could not be loaded."
    )

    st.write(
        "Error details:"
    )

    st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🍽️ AI Meal Planner | "
    "RAG Recipe Book + Generative AI"
)