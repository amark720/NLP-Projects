import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(llm, portfolio, clean_text):
    # Page title with emoji
    st.title("📧 Cold Mail Generator")

    # Input section
    st.markdown("### Enter Job Posting URL:")
    url_input = st.text_input(
        "",
        placeholder="Enter a URL like https://example.com/job-posting"
    )

    # Submit button styled
    submit_button = st.button("🚀 Generate Email")

    # Placeholder for loading or output
    output_placeholder = st.empty()

    if submit_button:
        with output_placeholder.container():
            # Show "Generating Cold Email..." while processing
            st.markdown(
                """
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 200px;
                    font-size: 26px;
                    font-weight: bold;
                    color: #E9D5BD;
                ">
                    Generating Cold Email...
                </div>
                """,
                unsafe_allow_html=True
            )

        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)

            # Clear loading text
            output_placeholder.empty()

            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)

                # Display the generated email
                st.success("✅ Here is Your Generated Email:")
                st.code(email, language='markdown')
        except Exception as e:
            # Clear loading text and show error
            output_placeholder.empty()
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)


