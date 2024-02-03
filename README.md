# Flask API for Similarity Search

This Flask application provides an API for performing similarity searches within a dataset using embeddings. The application uses OpenAI's API to generate embeddings for text inputs and compares these with a pre-computed embeddings dataset to find and return the most similar items. This application is intended for experimental purposes only and should not be used in production environments.

## Features

- **Environment Variable Management**: Utilizes `dotenv` for secure API key management.
- **OpenAI Integration**: Leverages OpenAI's API to create text embeddings.
- **Similarity Search**: Implements cosine similarity to find the most similar items based on embeddings.
- **Pandas and NumPy**: Uses Pandas for data manipulation and NumPy for numerical operations.

## Setup

### Requirements

- Python
- Flask
- Pandas
- NumPy
- openai
- python-dotenv

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the app directory:
   ```
   cd <app-directory>
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory of the application and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. Load the pre-computed embeddings CSV file (`data/<file-name>.csv`) into the application's data directory.

### Running the Application

To start the application, run:

```
python app.py
```

The application will be available at `http://localhost:5000`. Try your first request by navigating to `http://localhost:5000/sim_search/<text>` in your browser or using a tool like Postman.

## API Endpoints

### Similarity Search

- **URL**: `/sim_search/<text>`
- **Method**: `GET`
- **URL Params**: `text=[string]`
- **Success Response**: A JSON array of the top 5 most similar items based on the text input.
- **Error Response**: Error message in case of failure.

### Hyde Search (TODO)

- **URL**: `/hyde_search/<term>`
- **Method**: `GET`
- **Description**: This endpoint is planned for future implementation.

## Contributing

Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.
