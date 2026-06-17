# HOTALAR

A demo hotel review intelligence app. The backend (FastAPI) runs BERT-based sentiment and aspect-sentiment analysis on guest reviews and uses OpenAI to draft reply suggestions in three different tones. The frontend (Streamlit) provides an inbox for staff to browse reviews, generate replies, and view a dashboard of aggregate metrics.

To run locally: create a virtualenv, `pip install -r requirements.txt`, drop an `OPENAI_API_KEY` into a `.env` file, then start the backend with `uvicorn main:app --reload` and the frontend with `streamlit run streamlit_app.py` in a separate terminal. The first backend launch downloads the BERT models from HuggingFace (~1.4 GB) into a local `models/` folder. A 30-row `test_data.csv` is included for quick demo.
