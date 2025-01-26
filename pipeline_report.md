# Pipeline Report: CI/CD Pipeline for MLOps Assignment

## Overview
This report documents the CI/CD pipeline designed for the MLOps-Assignment project. The pipeline is implemented using GitHub Actions to ensure code quality, reliability, and automation of development workflows. It includes stages for linting, testing, and placeholder deployment.

---

## Pipeline Stages

### 1. Linting
**Purpose**:  
The linting stage ensures that the project adheres to Python's PEP 8 style guide. It helps maintain clean and consistent code.

**Tool Used**:  
- **flake8**: A Python linting tool to identify style violations, unused imports, and other code-quality issues.

**Configuration**:  
- The `flake8` tool runs on the `app` directory, checking all Python files for PEP 8 compliance.
- Relevant dependencies are listed in the `requirements.txt` file:
  ```plaintext
  flake8
  pytest
  flask
