# 📧 GenAI Cold Email Generator
Cold Email Generator is an AI-powered Email Generator designed to streamline the job application process using groq, langchain and streamlit. It allows users to input the URL of a company's careers page. The tool then extracts job listings from that page and generates personalized cold emails. These emails include relevant portfolio links sourced from a vector database, based on the specific job descriptions.


📍 **_Imagine a scenario:_**

- **EY’s Hiring Need:** EY is looking for a Principal Software Engineer and is investing time and resources in hiring, onboarding, and training processes.

- **FreshersGold’s Offering:** FreshersGold, a software development company, can provide a dedicated software development engineer to EY, simplifying their hiring process.

- **Sachin’s Task:** Sachin, a business development executive at FreshersGold, plans to reach out to EY via a cold email to propose their services.

- **Using the Cold Email Generator:**

  1. Sachin opens the Cold Email Generator app.
  2. He inputs the link to EY’s job description page.
  3. The app analyzes the job description, skills, and responsibilities to create a tailored cold email.
  4. Sachin sends the generated email directly to EY, saving significant time and effort.

**This process streamlines Sachin’s workflow, reduces manual effort, and ensures a professional, personalized email is sent to the company.**


### ScreenRecording of Live App:
<img src="https://github.com/amark720/NLP-Projects/blob/main/GenAI%20Cold%20Email%20Generator/ScreenRecording.gif" width=100% height=90% >

### Features:

- **Web Scraping:** Automatically extracts job details from posting URLs using WebBaseLoader.
- **AI-Powered Analysis:** Utilizes "llama-3.1-70b-versatile" to convert job data into structured JSON format.
- **Semantic Search:** Employs ChromaDB for efficient information retrieval.
- **Personalized Email Generation:** Creates tailored cold emails based on job specifics.
- **User-Friendly Interface:** Built with Streamlit for easy interaction.
- **Cloud-Based Processing:** Employs Groq cloud for rapid inference.

## Architecture Diagram
![img.png](imgs/architecture.png)

## Set-up
1. To get started we first need to get an API_KEY from here: https://console.groq.com/keys. Inside `app/.env` update the value of `GROQ_API_KEY` with the API_KEY you created. 


2. To get started, first install the dependencies using:
    ```commandline
     pip install -r requirements.txt
    ```
   
3. Run the streamlit app:
   ```commandline
   streamlit run app/main.py
   ```
   
### Two websites to find job links for reference:
1. <a href="https://jobs.nike.com/?jobSearch=true&jsKeywords%5B0%5D=&jsOffset=0&jsSort=posting_start_date&jsLanguage=en">NIKE Career Page!</a>  
2. <a href="https://careers.ey.com/search/?createNewAlert=false&q=&locationsearch=&optionsFacetsDD_country=&optionsFacetsDD_customfield1=">EY Career Page!</a>  

### Technologies Used:

Python, LangChain, llama-3.1-70b-versatile (Open-source model), Groq Cloud, Streamlit, ChromaDB, and WebBaseLoader.

#### 📧 Feel Free to contact me at➛ amark720@gmail.com for any help related to this Project!

<!-- 
Notes: We didn't used Ollama coz the inference of response was slow, so we used GroqCloud whose inference was fast coz it uses LPU(Language processing unit).
2. We used ChromaDb over traditional DB coz it does meaning based Sementic search so it gives results similar to user's query based on eucledian distance.(12:00)  
Reference- https://www.youtube.com/watch?v=CO4E_9V6li0&ab_channel=codebasics -->
