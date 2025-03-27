# Simple Chat Application

A simple chat application with a FastAPI backend and a HTML/CSS/JavaScript frontend.

## Project Structure

```
Week_8/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
└── README.md
```

## Setup and Running

### Backend

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```
   python main.py
   ```
   The server will start at `http://localhost:8000`

### Frontend

#### Installing Node.js and npm

Before using npm's serve package, you need to install Node.js and npm:

**For Windows:**
1. Download the Node.js installer from [nodejs.org](https://nodejs.org/)
2. Run the installer and follow the installation wizard
3. Verify installation by opening Command Prompt and typing:
   ```
   node -v
   npm -v
   ```

**For macOS:**
1. Using Homebrew:
   ```
   brew install node
   ```
   
   Or download the installer from [nodejs.org](https://nodejs.org/)
2. Verify installation:
   ```
   node -v
   npm -v
   ```

**For Linux (Ubuntu/Debian):**
1. Install using apt:
   ```
   sudo apt update
   sudo apt install nodejs npm
   ```
2. Verify installation:
   ```
   node -v
   npm -v
   ```

#### Using npm's serve package

1. Install serve globally (if not already installed):
   ```
   npm install -g serve
   ```

2. Navigate to the frontend directory:
   ```
   cd frontend
   ```

3. Run serve to host the static files:
   ```
   serve -s .
   ```
   By default, this will serve your frontend on port 5000 (http://localhost:5000).

4. You can specify a different port with the `-l` flag if needed:
   ```
   serve -s . -l 3000
   ```

## Usage

1. Type a message in the input field.
2. Press Enter or click the Send button.
3. The server will respond with a dummy message. 