# Contributing to SR University ChatBot

Thank you for your interest in contributing to the SR University ChatBot! This document provides guidelines and instructions for contributors.

## 🤝 Ways to Contribute

- **Bug Reports**: Report issues or bugs you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with fixes or enhancements
- **Documentation**: Improve documentation and examples
- **Testing**: Add tests or improve existing test coverage

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/University_ChatBot.git
cd University_ChatBot
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions small and focused

### Commit Messages

Use conventional commit format:
```
type(scope): description

feat(chat): add conversation history
fix(ui): resolve mobile responsiveness issue
docs(readme): update installation instructions
```

### File Structure

```
University_ChatBot/
├── app.py              # Main Flask application
├── model.py            # Alternative model implementation
├── templates/          # Jinja2 templates
├── static/            # CSS, JS, and images
├── faiss_index/       # Vector database files
├── tests/             # Test files (if added)
└── docs/              # Additional documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests (if test suite exists)
python -m pytest

# Run specific test file
python -m pytest tests/test_app.py

# Run with coverage
python -m pytest --cov=app
```

### Manual Testing

1. Start the application: `python app.py`
2. Navigate to `http://localhost:5000`
3. Test various query types:
   - University-specific questions
   - Edge cases and error handling
   - UI responsiveness

## 📝 Pull Request Process

### Before Submitting

1. **Test your changes thoroughly**
2. **Update documentation** if needed
3. **Ensure code follows style guidelines**
4. **Add comments for complex logic**

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Changes tested manually
- [ ] Documentation updated (if applicable)
- [ ] No breaking changes (or clearly documented)

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Manual testing completed
- [ ] All existing tests pass
- [ ] New tests added (if applicable)

## Screenshots (if applicable)
Add screenshots for UI changes.
```

## 🐛 Bug Reports

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version**
3. **Gather relevant information**

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the issue.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. iOS]
- Browser: [e.g. chrome, safari]
- Python version: [e.g. 3.8]
- Dependencies: [relevant package versions]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem.

**Describe the solution you'd like**
Clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features considered.

**Additional context**
Any other context, mockups, or screenshots.
```

## 🛠️ Development Setup

### Environment Variables

Required variables in `.env`:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
PINECONE_API_KEY=your_key_here  # Optional
```

### Adding New Features

1. **Plan the feature**: Discuss in issues first
2. **Design the API**: Consider backward compatibility
3. **Implement incrementally**: Small, focused commits
4. **Test thoroughly**: Both unit and integration tests
5. **Document**: Update README and code comments

### Working with the AI Pipeline

- **Models**: Located in `get_answer()` function
- **Embeddings**: Uses sentence-transformers
- **Vector DB**: FAISS for similarity search
- **Prompts**: Customizable in prompt templates

## 📚 Documentation

### Adding Documentation

- **README.md**: Main project documentation
- **Code comments**: Inline explanations
- **Docstrings**: Function and class documentation
- **API docs**: Endpoint documentation

### Documentation Style

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep information up-to-date

## 🔍 Code Review

### What We Look For

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it readable and maintainable?
- **Performance**: Are there any performance issues?
- **Security**: Are there any security concerns?
- **Testing**: Is it adequately tested?

### Review Process

1. **Automated checks**: CI/CD pipeline runs
2. **Manual review**: Code review by maintainers
3. **Testing**: Feature testing by reviewers
4. **Approval**: At least one maintainer approval required

## 🎯 Project Goals

- **Accuracy**: Provide correct university information
- **Performance**: Fast response times
- **Usability**: Intuitive user interface
- **Maintainability**: Clean, documented code
- **Scalability**: Handle increased usage

## 📞 Getting Help

- **Issues**: Create GitHub issues for questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers directly for sensitive issues

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the SR University ChatBot! 🎓✨