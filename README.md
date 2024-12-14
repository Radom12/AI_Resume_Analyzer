# AI Resume Analyzer

## Overview
The **AI Resume Analyzer** is a comprehensive web application developed using Streamlit that empowers users to gain insights into their resumes, identify skill gaps, and receive relevant course and job recommendations. This application leverages advanced NLP tools, resume parsing, and machine learning to provide users with personalized resume feedback, recommended fields, and job opportunities.

## Key Features
- **Resume Analysis**: Extracts and analyzes the content of resumes to highlight skills, education, and work experience.
- **Skill and Field Recommendations**: Suggests courses and fields based on the analyzed skills.
- **Job Search**: Integrates web scraping to help users find job listings tailored to their profile.
- **Feedback Collection**: Enables users to submit feedback, which is stored for continuous improvement.
- **Admin Dashboard**: Provides an admin interface for managing and visualizing user data and feedback, including analytics such as user activity and experience distribution.

## Technologies Used
- **Frontend**: Streamlit for user interface
- **Backend**: Python with libraries such as Pandas, Spacy, NLTK, Selenium, and Plotly
- **Database**: MySQL for storing user and feedback data
- **PDF Parsing**: pdfminer for text extraction from resumes
- **NLP**: Spacy and NLTK for text processing and entity recognition

## Installation
### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Radom12/AI_Resume_Analyzer
   cd AI_Resume_Analyzer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the database:
   - Ensure you have a MySQL server running.
   - Create a database named `resume_analyzer`.
   - Configure `.env` with your database credentials:
     ```
     DB_HOST=localhost
     DB_USER=root
     DB_PASSWORD=yourpassword
     DB_NAME=resume_analyzer
     ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
.
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── Courses.py             # Module containing recommended course data
├── README.md              # Project documentation
└── .env                   # Environment variables for database configuration
```

## Usage Guide
1. **Upload Your Resume**: Navigate to the 'User' section and upload a PDF resume to get detailed analysis and feedback.
2. **Find Jobs**: Use the 'Find Jobs' page to search for jobs based on title and location.
3. **Admin Access**: Log in to access the admin dashboard, where you can view user data, feedback, and various analytics.
4. **Feedback**: Provide Feedback on the app.
5. **Improvement**:Improve your resume based on the suggestions.

## Security and Privacy
Your resume data is only used during your session for analysis and is not permanently stored.

### Known Issues
1. **Slow Job Finding Process**: The job search feature using web scraping can be slow, affecting user experience. Optimization efforts are in progress to improve response times.
2. **Improper Admin Visualization**: The admin dashboard may show inconsistent or unclear visual representations of user data. Enhancements to data visualization are planned for the next update.
3. **Limited Feedback Options**: The current feedback system provides basic text input, limiting user interaction. An expanded feedback mechanism with more structured input options will be implemented soon.
These issues are actively being worked on, and improvements will be released in future updates to enhance functionality and user experience.

## Star History

<a href="https://star-history.com/#Radom12/AI_Resume_Analyzer&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Radom12/AI_Resume_Analyzer&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Radom12/AI_Resume_Analyzer&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Radom12/AI_Resume_Analyzer&type=Date" />
 </picture>
</a>
