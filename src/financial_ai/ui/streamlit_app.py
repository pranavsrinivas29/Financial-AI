import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Financial AI",
    page_icon="📈",
    layout="wide",
)


st.title("Financial AI Assistant")

st.caption(
    "ML predictions + news sentiment + SEC RAG"
)


ticker = st.text_input(
    "Ticker",
    value="AAPL",
)


as_of_date = st.date_input(
    "As-of date",
)


query = st.text_area(
    "Question",
    value=(
        "Analyze Apple using ML, "
        "news sentiment and SEC filings"
    ),
    height=120,
)


if st.button("Analyze"):

    payload = {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date.isoformat(),
        "query": query,
    }

    with st.spinner(
        "Running financial analysis..."
    ):

        try:

            response = requests.post(
                f"{API_URL}/analyze",
                json=payload,
                timeout=600,
            )

            response.raise_for_status()

            result = response.json()

        except requests.RequestException as exc:

            st.error(
                f"API request failed: {exc}"
            )

            st.stop()


    st.success(
        "Analysis completed"
    )


    st.subheader(
        "Final Analysis"
    )

    st.write(
        result.get(
            "final_answer",
            "No final answer returned.",
        )
    )


    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "ML Outlook"
        )

        ml_result = result.get(
            "ml_result"
        )

        if ml_result:

            direction = (
                ml_result
                .get(
                    "direction",
                    {},
                )
            )

            volatility = (
                ml_result
                .get(
                    "volatility",
                    {},
                )
            )

            probability = (
                direction.get(
                    "positive_probability"
                )
            )

            predicted_class = (
                direction.get(
                    "predicted_class"
                )
            )

            expected_volatility = (
                volatility.get(
                    "expected_annualized_volatility"
                )
            )


            if probability is not None:

                st.metric(
                    "Positive Probability",
                    f"{probability:.2%}",
                )


            if predicted_class:

                st.metric(
                    "Predicted Direction",
                    predicted_class.title(),
                )


            if expected_volatility is not None:

                st.metric(
                    "Expected Annualized Volatility",
                    f"{expected_volatility:.2%}",
                )


            st.caption(
                "Feature date: "
                f"{ml_result.get('feature_date')}"
            )


            st.caption(
                "Training cutoff: "
                f"{ml_result.get('training_cutoff')}"
            )

        else:

            st.info(
                "ML agent was not used "
                "or returned no result."
            )


    with col2:

        st.subheader(
            "News Sentiment"
        )

        nlp_result = result.get(
            "nlp_result"
        )

        if nlp_result:

            features = (
                nlp_result
                .get(
                    "features",
                    {},
                )
            )


            st.metric(
                "Articles",
                nlp_result.get(
                    "article_count",
                    0,
                ),
            )


            st.metric(
                "Positive",
                features.get(
                    "positive_news_count_7d",
                    0,
                ),
            )


            st.metric(
                "Neutral",
                features.get(
                    "neutral_news_count_7d",
                    0,
                ),
            )


            st.metric(
                "Negative",
                features.get(
                    "negative_news_count_7d",
                    0,
                ),
            )


            sentiment_score = (
                features.get(
                    "news_sentiment_score_7d"
                )
            )


            if sentiment_score is not None:

                st.metric(
                    "Sentiment Score",
                    f"{sentiment_score:.3f}",
                )

        else:

            st.info(
                "NLP agent was not used "
                "or returned no result."
            )


    st.divider()


    st.subheader(
        "SEC RAG Evidence"
    )

    rag_result = result.get(
        "rag_result"
    )

    if rag_result:

        rag_answer = (
            rag_result.get(
                "answer"
            )
        )

        if rag_answer:

            st.write(
                rag_answer
            )


        sources = (
            rag_result.get(
                "sources",
                [],
            )
        )

        if sources:

            with st.expander(
                "Sources"
            ):

                for source in sources:

                    st.json(
                        source
                    )

    else:

        st.info(
            "RAG agent was not used "
            "or returned no result."
        )


    st.divider()


    errors = result.get(
        "errors",
        {},
    )

    if any(
        errors.values()
    ):

        with st.expander(
            "Agent Errors"
        ):

            st.json(
                errors
            )


    with st.expander(
        "Raw API Response"
    ):

        st.json(
            result
        )