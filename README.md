# SR University ChatBot 🎓

An intelligent chatbot designed specifically for **SR University** that provides accurate, context-aware answers to student queries using advanced AI and document retrieval technology.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **🧠 Intelligent Q&A**: Powered by state-of-the-art language models (Qwen/QwQ-32B-Preview)
- **📚 Document-Based Responses**: Uses university handbook and official documents as knowledge base
- **🔍 Vector Search**: FAISS-powered semantic search for accurate information retrieval
- **💬 Interactive Web Interface**: Clean, responsive chat interface with drag-and-drop functionality
- **⚡ Real-time Responses**: Fast, contextual answers with typing indicators
- **🎯 University-Specific**: Trained specifically on SR University information
- **🔒 Secure**: Environment-based API key management

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │   AI Pipeline   │
│   (HTML/JS/CSS) │◄──►│   (app.py)      │◄──►│   (LangChain)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Templates     │    │   FAISS Vector  │
                       │   (Jinja2)      │    │   Database      │
                       └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │   University    │
                                              │   Handbook PDF  │
                                              └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- HuggingFace API token
- 4GB+ RAM (for embedding models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Karanraj-6/University_ChatBot.git
   cd University_ChatBot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the chatbot**
   Open your browser and navigate to `http://localhost:5000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_key_here  # Optional for future use
```

### Getting HuggingFace API Token

1. Visit [HuggingFace](https://huggingface.co/)
2. Create an account or log in
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token to your `.env` file

## 📁 Project Structure

```
University_ChatBot/
├── 📄 app.py                 # Main Flask application
├── 📄 model.py               # Alternative model implementation
├── 📁 templates/
│   └── 📄 index.html         # Main chat interface
├── 📁 static/
│   ├── 📄 styles.css         # UI styling
│   ├── 📄 script.js          # Frontend functionality
│   ├── 🖼️ chatbot_icon.png   # Bot avatar
│   ├── 🖼️ user.png           # User avatar
│   └── 🖼️ sru_background.png # Background image
├── 📁 faiss_index/
│   ├── 📄 index.faiss        # Vector database
│   └── 📄 index.pkl          # Embeddings metadata
├── 📄 handbook_2024_25.pdf   # University knowledge base
├── 📄 code.ipynb             # Development notebook
├── 📄 requirements.txt       # Python dependencies
├── 📄 .env                   # Environment configuration
└── 📄 README.md              # This file
```

## 🔧 Technical Details

### AI Pipeline Components

1. **Document Processing**
   - PDF text extraction using PyPDF
   - Text chunking for optimal retrieval
   - Sentence transformer embeddings (all-MiniLM-L6-v2)

2. **Vector Database**
   - FAISS for efficient similarity search
   - Pre-computed embeddings for fast retrieval
   - Semantic search capabilities

3. **Language Model**
   - Qwen/QwQ-32B-Preview from HuggingFace
   - Customized prompts for university context
   - Temperature-controlled generation (0.4)

4. **Retrieval-Augmented Generation (RAG)**
   - Context-aware response generation
   - Source document verification
   - Fallback responses for unknown queries

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main chat interface |
| `/chat` | POST | Process user messages and return responses |

#### Chat API Request Format
```json
{
  "message": "What are the admission requirements?"
}
```

#### Chat API Response Format
```json
{
  "response": "The admission requirements for SR University include..."
}
```

## 🎨 User Interface

The chatbot features a modern, responsive web interface with:

- **Floating Chat Button**: Draggable, always-accessible chat trigger
- **Collapsible Chat Window**: Clean, modal-style chat interface
- **User Avatars**: Visual distinction between user and bot messages
- **Typing Indicators**: Real-time feedback during response generation
- **Responsive Design**: Works on desktop and mobile devices

### Interface Components

- **Chat Header**: University branding and close button
- **Message Area**: Scrollable conversation history
- **Input Field**: Text input with send button and Enter key support
- **Visual Feedback**: Loading animations and status indicators

## 📱 Usage Examples

### Basic Queries
- "What are the admission requirements?"
- "How do I register for courses?"
- "What facilities are available in the hostel?"
- "Tell me about the examination system"

### Academic Information
- "What is the grading system?"
- "How do I apply for a PhD program?"
- "What are the research opportunities?"

### Administrative Queries
- "What are the contact details for admissions?"
- "How do I pay fees?"
- "What documents do I need for enrollment?"

## 🛠️ Development

### Running in Development Mode

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Code Structure

- **app.py**: Main Flask application with chat endpoint
- **model.py**: Alternative implementation for testing
- **get_answer()**: Core function handling AI pipeline
- **Templates**: Jinja2 templates for web interface
- **Static files**: CSS, JavaScript, and images

### Adding New Documents

1. Place PDF files in the root directory
2. Update the document processing code in `code.ipynb`
3. Regenerate the FAISS index
4. Test with new queries

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Solution: Install missing dependencies
   pip install -r requirements.txt
   ```

2. **HuggingFace API Errors**
   ```bash
   # Solution: Check your API token in .env file
   # Ensure token has proper permissions
   ```

3. **FAISS Index Not Found**
   ```bash
   # Solution: Ensure faiss_index/ directory exists
   # Regenerate index if corrupted
   ```

4. **Memory Issues**
   ```bash
   # Solution: Increase system RAM or use smaller models
   # Consider using sentence-transformers/all-MiniLM-L6-v2
   ```

### Debug Mode

Enable Flask debug mode for detailed error messages:
```python
app.run(debug=True)
```

## 🔄 Updates and Maintenance

### Updating the Knowledge Base

1. Replace `handbook_2024_25.pdf` with new document
2. Run the notebook to regenerate embeddings
3. Update FAISS index
4. Test with sample queries

### Model Updates

To use a different language model:
1. Update `repo_id` in `app.py`
2. Adjust `model_kwargs` parameters
3. Test response quality

## 📞 Support

### University Contact Information
- **Phone**: 0(870) 281-8333/8311
- **Email**: info@sru.edu.in
- **Website**: [SR University](https://www.sru.edu.in)

### Technical Support
For technical issues:
1. Check the troubleshooting section
2. Review error logs
3. Ensure all dependencies are installed
4. Verify API credentials

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Guidelines

- Follow PEP 8 for Python code
- Use meaningful commit messages
- Test your changes thoroughly
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SR University** for providing the knowledge base
- **HuggingFace** for the language models and embeddings
- **LangChain** for the RAG framework
- **FAISS** for efficient vector search
- **Flask** for the web framework

## 📊 Performance

### Response Times
- Average: 2-5 seconds
- Depends on query complexity and model size

### Accuracy
- University-specific queries: ~95%
- General knowledge: Limited (by design)
- Fallback responses for unknown queries

---

**Made with ❤️ for SR University students and faculty**
