# Auto Dashboard Agent

This project is an automated query-to-dashboard generator for DP World, using Flask, Streamlit, OpenAI, and Vertica.

## Features
- Natural language to dashboard: Enter a business question, get a live dashboard.
- Uses OpenAI LLMs for table selection, KPI planning, and SQL generation.
- Connects to Vertica for real data.
- Modular agents for easy extension.

## Project Structure
- `app.py` — Flask web server
- `config.py` — Loads environment variables and configuration
- `requirements.txt` — Python dependencies
- `test/` — Unit tests
- `utils/` — Utility modules (e.g., logging)
- `logs/` — Log files
- `templates/` — HTML templates
- `static/` — Static assets (images, CSS, etc.)

## Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/Nayab-zak/auto_dashboard_agent.git
   cd auto_dashboard_agent
   ```
2. Create and activate your Python environment (conda or venv recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your secrets and DB info:
   ```bash
   cp .env.example .env
   # Edit .env with your keys and credentials
   ```
5. Run the app:
   ```bash
   python app.py
   ```

## Environment Variables
See `.env.example` for all required variables.

## Testing
Run all tests with:
```bash
python -m unittest discover -s test
```

## Security
**Never commit your real `.env` file or secrets to a public repo!**

## License
Proprietary / Internal
