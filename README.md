# Soupy Bot

An intelligent social media bot powered by LM Studio that generates contextual posts and replies. Primarily designed for Bluesky with advanced features like thread context analysis, URL enrichment, and smart content curation.

## Overview

Soupy Bot uses local LLM models via LM Studio to create human-like social media content. It intelligently analyzes posts, extracts context from linked articles, and generates relevant responses. The bot is optimized for Bluesky's AT Protocol API with comprehensive thread understanding and content enrichment.

## Key Features

### 🧠 **Intelligent Content Generation**
- Local LLM integration via LM Studio (OpenAI-compatible API)
- Context-aware post analysis and reply generation
- Smart topic extraction and search query generation
- Natural language processing for thread context

### 🔗 **Advanced Context Enrichment**
- Automatic URL content extraction and summarization
- Web search integration for additional context
- Thread traversal to understand conversation history
- Multi-source information synthesis

### 📱 **Bluesky Integration**
- Full AT Protocol API support (read/write)
- Thread context gathering and analysis
- Rich link previews with external embeds
- Author cooldown and deduplication systems
- Smart post selection from trending/popular feeds

### 🎯 **Smart Content Curation**
- Diverse seed post selection (author and topic diversity)
- Multi-factor article scoring (quality, reputation, content)
- Clickbait detection and filtering
- Domain reputation weighting

## Quick Start

### Prerequisites
- Python 3.10+
- LM Studio running locally with REST API enabled
- Bluesky account with app password

### Installation
```bash
# Clone and install dependencies
git clone <repository-url>
cd soupy-twitter
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Configuration (.env)
```bash
# LM Studio Configuration
OPENAI_BASE_URL=http://127.0.0.1:5112/v1
LOCAL_KEY=lm-studio
LOCAL_CHAT=your-model-name

# Bluesky Configuration
USE_BSKY=true
BSKY_HANDLE=yourhandle.bsky.social
BSKY_APP_PASSWORD=your-app-password

# Bot Personality
BEHAVIOUR="your personality prompt here"

# Scheduling
ACTIONS_PER_HOUR_CAP=4
MIN_INTERVAL_MINUTES=5
MAX_INTERVAL_MINUTES=30
```

### Running the Bot
```bash
# Start continuous operation
python main.py

# Force immediate actions
python main.py --postnow    # Generate and post daily news
python main.py --now --reply # Reply to a post immediately
```

## Usage Modes

### Daily News Posts (`--postnow`)
- Analyzes trending Bluesky posts using LLM
- Extracts URLs and uses their content for context
- Generates intelligent search queries
- Curates high-quality news articles
- Creates contextual commentary with rich link previews

### Reply Mode
- Reads Bluesky timeline and popular feeds
- Selects diverse, high-quality posts
- Analyzes thread context and conversation history
- Generates contextual replies with personality
- Includes URL enrichment when available

### Content Enrichment
- **URL Analysis**: Fetches and summarizes linked articles
- **Web Search**: Derives additional context from web searches
- **Thread Context**: Understands conversation flow and history
- **Smart Filtering**: Avoids clickbait and low-quality content

## Advanced Features

### Thread Context Analysis
The bot automatically traverses thread structures to understand conversation context:
- Identifies root posts and conversation depth
- Analyzes ancestor posts for context
- Considers sibling replies for conversation tone
- Includes child replies for response appropriateness

### Intelligent Post Selection
- **Author Diversity**: Limits posts per author to ensure variety
- **Content Diversity**: Analyzes topic keywords to avoid repetition
- **Quality Scoring**: Multi-factor scoring including engagement, recency, and content quality
- **Smart Filtering**: Excludes self-posts and recently replied content

### Article Curation
- **Multi-Factor Scoring**: Content length, title quality, domain reputation
- **Clickbait Detection**: Filters out sensationalist headlines
- **Quality Indicators**: Rewards analysis, reports, studies, and data
- **Domain Reputation**: Prioritizes trusted news sources

## Configuration Options

### Core Settings
- `BEHAVIOUR`: System prompt defining bot personality
- `ACTIONS_PER_HOUR_CAP`: Rate limiting (default: 4)
- `MIN_INTERVAL_MINUTES` / `MAX_INTERVAL_MINUTES`: Action timing
- `OWN_POSTING_PROBABILITY`: Balance between posts and replies

### Enrichment Controls
- `URL_ENRICHMENT`: Enable/disable URL content analysis
- `WEB_SEARCH_ENRICHMENT`: Enable/disable web search context
- `ENRICHMENT_MAX_CHARS`: Maximum context length
- `WEB_SEARCH_RESULTS`: Number of search results to analyze

### Bluesky Settings
- `BSKY_AUTHOR_COOLDOWN_MINUTES`: Time between replies to same author
- `BSKY_CANDIDATE_POOL_SIZE`: Number of posts to consider
- `BSKY_REPLIED_LOG_MAX`: Maximum reply history to track
- `BSKY_POST_MAX_CHARS`: Maximum post length (300 char limit)

## Future Roadmap

### Web Interface
- Command and control dashboard for bot management
- Real-time monitoring of bot activity and performance
- Configuration management through web UI
- Analytics and insights on bot interactions

### Discord Integration
- Discord bot for triggering posts and replies
- Real-time notifications of bot activity
- Manual override capabilities
- Community interaction features

### Enhanced Features
- Multi-platform support expansion
- Advanced analytics and reporting
- Custom personality training
- Community-driven content curation

## Technical Details

### Architecture
- **LLM Client**: Handles all language model interactions
- **Context Enricher**: Manages URL and web search enrichment
- **Bluesky Bot**: AT Protocol API integration and post management
- **Web Search**: DuckDuckGo integration for content discovery
- **Scheduler**: Intelligent timing and rate limiting

### API Integration
- **AT Protocol**: Official Bluesky API for all operations
- **LM Studio**: Local LLM inference via OpenAI-compatible API
- **DuckDuckGo**: Web search for content discovery and enrichment
- **TinyURL**: URL shortening for character optimization

## Safety and Compliance

- Uses official APIs only (no browser automation)
- Respects platform rate limits and terms of service
- Implements intelligent cooldowns and deduplication
- Focuses on quality content and meaningful interactions
- Designed for educational and personal use

## Contributing

This project is designed for educational purposes. Contributions are welcome for:
- Additional platform integrations
- Enhanced context analysis
- Improved content curation algorithms
- Web interface development
- Documentation improvements

## License

Educational use only. Please respect platform terms of service and use responsibly.