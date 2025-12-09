"""
Job Description Technology Extractor
=====================================
A comprehensive tool to extract technologies, skills, and keywords from job descriptions.
Works with JSON input data and provides multiple extraction methods.

Features:
- JSON parsing and manipulation
- Keyword-based extraction with variations
- NLP-based extraction using regex patterns
- Skill categorization by domain
- Statistical analysis of extracted technologies
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set, Tuple
import string


# =============================================================================
# TECHNOLOGY KNOWLEDGE BASE
# =============================================================================

TECHNOLOGY_DATABASE = {
    # Programming Languages
    "languages": {
        "Python": ["python", "py"],
        "Java": ["java", "jdk", "jre", "j2ee", "jvm"],
        "JavaScript": ["javascript", "js", "ecmascript", "es6", "es2015"],
        "TypeScript": ["typescript", "ts"],
        "C++": ["c++", "cpp", "cplusplus"],
        "C#": ["c#", "csharp", "c sharp", ".net c#"],
        "Go": ["golang", "go lang"],
        "Rust": ["rust", "rustlang"],
        "Ruby": ["ruby"],
        "PHP": ["php"],
        "Swift": ["swift"],
        "Kotlin": ["kotlin"],
        "Scala": ["scala"],
        "R": ["r programming", "r language", "rstudio"],
        "SQL": ["sql", "structured query language"],
        "Bash": ["bash", "shell script", "shell scripting"],
    },
    
    # Web Frameworks
    "web_frameworks": {
        "Django": ["django"],
        "Flask": ["flask"],
        "FastAPI": ["fastapi", "fast api"],
        "React": ["react", "reactjs", "react.js"],
        "Angular": ["angular", "angularjs", "angular.js"],
        "Vue.js": ["vue", "vuejs", "vue.js"],
        "Node.js": ["nodejs", "node.js", "node js"],
        "Express.js": ["express", "expressjs", "express.js"],
        "Spring": ["spring", "spring boot", "springboot"],
        "Ruby on Rails": ["rails", "ruby on rails", "ror"],
        "ASP.NET": ["asp.net", "aspnet", ".net core", "dotnet"],
        "Laravel": ["laravel"],
        "Next.js": ["nextjs", "next.js"],
        "Svelte": ["svelte", "sveltekit"],
    },
    
    # Databases
    "databases": {
        "MySQL": ["mysql", "my sql"],
        "PostgreSQL": ["postgresql", "postgres", "psql"],
        "MongoDB": ["mongodb", "mongo"],
        "SQLite": ["sqlite"],
        "Oracle": ["oracle", "oracle db", "oracle database"],
        "SQL Server": ["sql server", "mssql", "microsoft sql server"],
        "Redis": ["redis"],
        "Cassandra": ["cassandra"],
        "DynamoDB": ["dynamodb", "dynamo db"],
        "Elasticsearch": ["elasticsearch", "elastic search", "es"],
        "Firebase": ["firebase", "firestore"],
        "Neo4j": ["neo4j"],
        "CouchDB": ["couchdb", "couch db"],
        "MariaDB": ["mariadb"],
    },
    
    # Cloud Platforms & Services
    "cloud": {
        # AWS
        "AWS": ["aws", "amazon web services"],
        "Amazon EC2": ["ec2", "elastic compute cloud"],
        "Amazon S3": ["s3", "simple storage service"],
        "AWS Lambda": ["lambda", "aws lambda"],
        "Amazon RDS": ["rds", "relational database service"],
        "Amazon Redshift": ["redshift"],
        "AWS Glue": ["glue", "aws glue"],
        "Amazon EMR": ["emr", "elastic mapreduce"],
        "Amazon Kinesis": ["kinesis"],
        "Amazon SageMaker": ["sagemaker", "sage maker"],
        "AWS IAM": ["iam", "identity access management"],
        "Amazon CloudWatch": ["cloudwatch", "cloud watch"],
        "AWS CloudFormation": ["cloudformation", "cloud formation"],
        "Amazon Athena": ["athena"],
        "AWS Step Functions": ["step functions"],
        "Amazon SQS": ["sqs", "simple queue service"],
        "Amazon SNS": ["sns", "simple notification service"],
        "AWS ECS": ["ecs", "elastic container service"],
        "AWS EKS": ["eks", "elastic kubernetes service"],
        
        # Azure
        "Azure": ["azure", "microsoft azure"],
        "Azure Blob Storage": ["blob storage", "azure blob"],
        "Azure Data Lake": ["data lake storage", "adls"],
        "Azure SQL Database": ["azure sql"],
        "Azure Cosmos DB": ["cosmos db", "cosmosdb"],
        "Azure Synapse": ["synapse analytics", "azure synapse"],
        "Azure Functions": ["azure functions"],
        "Azure Data Factory": ["data factory", "adf"],
        "Azure DevOps": ["azure devops", "ado"],
        "Azure HDInsight": ["hdinsight"],
        "Power BI": ["power bi", "powerbi"],
        
        # Google Cloud
        "Google Cloud": ["gcp", "google cloud platform", "google cloud"],
        "BigQuery": ["bigquery", "big query"],
        "Cloud Storage": ["cloud storage", "gcs"],
        "Cloud Functions": ["cloud functions", "gcf"],
        "Dataflow": ["dataflow", "cloud dataflow"],
        "Dataproc": ["dataproc", "cloud dataproc"],
        "Pub/Sub": ["pubsub", "pub/sub", "cloud pub/sub"],
        "Cloud Composer": ["cloud composer", "composer"],
        "Vertex AI": ["vertex ai", "vertex"],
        "Cloud SQL": ["cloud sql"],
        "Bigtable": ["bigtable", "big table"],
    },
    
    # Data Engineering & Big Data
    "data_engineering": {
        "Apache Spark": ["spark", "pyspark", "apache spark"],
        "Apache Kafka": ["kafka", "apache kafka"],
        "Apache Hadoop": ["hadoop", "hdfs", "mapreduce"],
        "Apache Airflow": ["airflow", "apache airflow"],
        "Apache Hive": ["hive", "apache hive"],
        "Apache Flink": ["flink", "apache flink"],
        "Apache NiFi": ["nifi", "apache nifi"],
        "Snowflake": ["snowflake"],
        "Databricks": ["databricks"],
        "dbt": ["dbt", "data build tool"],
        "Informatica": ["informatica"],
        "Talend": ["talend"],
        "ETL": ["etl", "extract transform load"],
        "Data Pipeline": ["data pipeline", "data pipelines"],
        "Data Warehouse": ["data warehouse", "dwh", "data warehousing"],
    },
    
    # Machine Learning & AI
    "ml_ai": {
        "TensorFlow": ["tensorflow", "tf"],
        "PyTorch": ["pytorch", "torch"],
        "Scikit-Learn": ["scikit-learn", "sklearn", "scikit learn"],
        "Keras": ["keras"],
        "OpenCV": ["opencv", "open cv"],
        "NLTK": ["nltk"],
        "spaCy": ["spacy"],
        "Hugging Face": ["hugging face", "huggingface", "transformers"],
        "XGBoost": ["xgboost"],
        "LightGBM": ["lightgbm"],
        "MLflow": ["mlflow", "ml flow"],
        "Kubeflow": ["kubeflow"],
        "Deep Learning": ["deep learning", "neural network", "neural networks"],
        "NLP": ["nlp", "natural language processing"],
        "Computer Vision": ["computer vision", "cv", "image recognition"],
        "LLM": ["llm", "large language model", "gpt", "chatgpt"],
    },
    
    # Data Analysis & Visualization
    "data_analysis": {
        "Pandas": ["pandas"],
        "NumPy": ["numpy", "np"],
        "Matplotlib": ["matplotlib", "pyplot"],
        "Seaborn": ["seaborn"],
        "Plotly": ["plotly"],
        "Tableau": ["tableau"],
        "Power BI": ["power bi", "powerbi"],
        "Looker": ["looker"],
        "Metabase": ["metabase"],
        "Superset": ["superset", "apache superset"],
        "Excel": ["excel", "ms excel", "microsoft excel"],
        "Statistics": ["statistics", "statistical analysis"],
    },
    
    # DevOps & Infrastructure
    "devops": {
        "Docker": ["docker", "containerization"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Jenkins": ["jenkins"],
        "Git": ["git", "github", "gitlab", "bitbucket"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
        "Linux": ["linux", "ubuntu", "centos", "debian"],
        "Nginx": ["nginx"],
        "Apache": ["apache", "httpd"],
        "Prometheus": ["prometheus"],
        "Grafana": ["grafana"],
        "ELK Stack": ["elk", "elasticsearch logstash kibana", "kibana", "logstash"],
        "Helm": ["helm", "helm charts"],
        "ArgoCD": ["argocd", "argo cd"],
    },
    
    # Frontend Technologies
    "frontend": {
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3", "scss", "sass", "less"],
        "Bootstrap": ["bootstrap"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "jQuery": ["jquery"],
        "Webpack": ["webpack"],
        "Vite": ["vite"],
        "Redux": ["redux"],
        "GraphQL": ["graphql"],
        "REST API": ["rest", "restful", "rest api"],
    },
    
    # Testing
    "testing": {
        "Selenium": ["selenium"],
        "Jest": ["jest"],
        "PyTest": ["pytest"],
        "JUnit": ["junit"],
        "Cypress": ["cypress"],
        "Postman": ["postman"],
        "SonarQube": ["sonarqube", "sonar"],
        "Unit Testing": ["unit test", "unit testing"],
        "Integration Testing": ["integration test", "integration testing"],
    },
    
    # Other Tools & Concepts
    "other": {
        "Agile": ["agile", "scrum", "kanban"],
        "Jira": ["jira"],
        "Confluence": ["confluence"],
        "Slack": ["slack"],
        "API Development": ["api", "api development", "api design"],
        "Microservices": ["microservices", "microservice architecture"],
        "OAuth": ["oauth", "oauth2"],
        "JWT": ["jwt", "json web token"],
        "WebSocket": ["websocket", "web socket"],
    }
}


# =============================================================================
# JSON UTILITIES
# =============================================================================

class JSONHandler:
    """Handles JSON parsing, manipulation, and serialization."""
    
    @staticmethod
    def load_from_string(json_string: str) -> Dict:
        """Parse JSON from string."""
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict:
        """Load JSON from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_to_file(data: Any, filepath: str, indent: int = 2) -> None:
        """Save data to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def pretty_print(data: Any, indent: int = 2) -> str:
        """Return pretty-printed JSON string."""
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(JSONHandler.flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def extract_all_text(data: Any) -> str:
        """Extract all text content from nested JSON structure."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return ' '.join(JSONHandler.extract_all_text(v) for v in data.values())
        elif isinstance(data, list):
            return ' '.join(JSONHandler.extract_all_text(item) for item in data)
        else:
            return str(data) if data else ''


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
    """Handles text cleaning and preprocessing."""
    
    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'can', 'just', 'should', 'now', 'also',
        'into', 'our', 'their', 'your', 'we', 'you', 'them', 'him', 'her',
        'would', 'could', 'shall', 'may', 'might', 'must', 'need', 'etc',
        'able', 'about', 'across', 'after', 'almost', 'along', 'already',
        'although', 'always', 'among', 'any', 'anyone', 'anything', 'anywhere',
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve important tech symbols before cleaning
        text = text.replace('c++', 'cplusplus')
        text = text.replace('c#', 'csharp')
        text = text.replace('.net', 'dotnet')
        text = text.replace('node.js', 'nodejs')
        text = text.replace('vue.js', 'vuejs')
        text = text.replace('react.js', 'reactjs')
        text = text.replace('express.js', 'expressjs')
        text = text.replace('next.js', 'nextjs')
        text = text.replace('ci/cd', 'cicd')
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def remove_stopwords(text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        return ' '.join(w for w in words if w not in TextPreprocessor.STOPWORDS)
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        return text.split()
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text."""
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams


# =============================================================================
# TECHNOLOGY EXTRACTOR
# =============================================================================

class TechnologyExtractor:
    """Main class for extracting technologies from job descriptions."""
    
    def __init__(self, tech_database: Dict = None):
        """Initialize with optional custom technology database."""
        self.tech_database = tech_database or TECHNOLOGY_DATABASE
        self._build_lookup_table()
    
    def _build_lookup_table(self) -> None:
        """Build efficient lookup table for technology matching."""
        self.lookup_table = {}
        for category, technologies in self.tech_database.items():
            for tech_name, variations in technologies.items():
                for variation in variations:
                    key = variation.lower()
                    if key not in self.lookup_table:
                        self.lookup_table[key] = []
                    self.lookup_table[key].append({
                        'name': tech_name,
                        'category': category
                    })
    
    def extract_from_text(self, text: str, include_category: bool = True) -> Dict[str, Any]:
        """Extract technologies from a single text string."""
        cleaned_text = TextPreprocessor.clean_text(text)
        
        found_technologies = {}
        matched_positions = []
        
        # Check each technology variation
        for variation, tech_list in self.lookup_table.items():
            # Use word boundary matching for accuracy
            pattern = r'\b' + re.escape(variation) + r'\b'
            matches = list(re.finditer(pattern, cleaned_text))
            
            if matches:
                for tech_info in tech_list:
                    tech_name = tech_info['name']
                    category = tech_info['category']
                    
                    if tech_name not in found_technologies:
                        found_technologies[tech_name] = {
                            'category': category,
                            'count': len(matches),
                            'matched_variations': [variation]
                        }
                    else:
                        found_technologies[tech_name]['count'] += len(matches)
                        if variation not in found_technologies[tech_name]['matched_variations']:
                            found_technologies[tech_name]['matched_variations'].append(variation)
        
        # Organize by category if requested
        if include_category:
            categorized = defaultdict(list)
            for tech_name, info in found_technologies.items():
                categorized[info['category']].append({
                    'name': tech_name,
                    'count': info['count'],
                    'matched_variations': info['matched_variations']
                })
            return dict(categorized)
        
        return found_technologies
    
    def extract_from_job_description(self, job_data: Dict) -> Dict[str, Any]:
        """Extract technologies from a job description dictionary."""
        # Combine all text fields
        text_fields = []
        
        # Extract description
        if 'description' in job_data:
            text_fields.append(job_data['description'])
        
        # Extract skills (handle nested structure)
        if 'skills' in job_data:
            skills = job_data['skills']
            if isinstance(skills, dict):
                for key, value in skills.items():
                    if isinstance(value, list):
                        text_fields.extend(value)
                    else:
                        text_fields.append(str(value))
            elif isinstance(skills, list):
                text_fields.extend(skills)
        
        # Combine all text
        combined_text = ' '.join(text_fields)
        
        # Extract technologies
        extracted = self.extract_from_text(combined_text)
        
        return {
            'job_id': job_data.get('job_id'),
            'role': job_data.get('role'),
            'location': job_data.get('location'),
            'experience': job_data.get('experience'),
            'extracted_technologies': extracted
        }
    
    def extract_from_json(self, json_data: Dict) -> List[Dict]:
        """Extract technologies from JSON containing multiple job descriptions."""
        results = []
        
        # Handle different JSON structures
        if 'job_descriptions' in json_data:
            jobs = json_data['job_descriptions']
        elif isinstance(json_data, list):
            jobs = json_data
        else:
            jobs = [json_data]
        
        for job in jobs:
            result = self.extract_from_job_description(job)
            results.append(result)
        
        return results
    
    def get_technology_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from extraction results."""
        all_technologies = Counter()
        category_counts = defaultdict(Counter)
        tech_by_role = defaultdict(Counter)
        
        for result in results:
            role = result.get('role', 'Unknown')
            extracted = result.get('extracted_technologies', {})
            
            for category, tech_list in extracted.items():
                for tech in tech_list:
                    tech_name = tech['name']
                    count = tech['count']
                    
                    all_technologies[tech_name] += count
                    category_counts[category][tech_name] += count
                    tech_by_role[role][tech_name] += count
        
        return {
            'total_jobs_analyzed': len(results),
            'unique_technologies_found': len(all_technologies),
            'top_technologies': all_technologies.most_common(20),
            'technologies_by_category': {
                cat: dict(counter.most_common(10))
                for cat, counter in category_counts.items()
            },
            'technologies_by_role': {
                role: dict(counter.most_common(10))
                for role, counter in tech_by_role.items()
            }
        }


# =============================================================================
# SKILL MATCHER (for comparing resume to job description)
# =============================================================================

class SkillMatcher:
    """Match skills between job descriptions and candidate profiles."""
    
    def __init__(self, extractor: TechnologyExtractor = None):
        self.extractor = extractor or TechnologyExtractor()
    
    def match_skills(self, job_text: str, candidate_skills: List[str]) -> Dict[str, Any]:
        """Compare job requirements with candidate skills."""
        # Extract job requirements
        job_tech = self.extractor.extract_from_text(job_text, include_category=False)
        job_skills = set(job_tech.keys())
        
        # Normalize candidate skills
        candidate_normalized = set()
        for skill in candidate_skills:
            extracted = self.extractor.extract_from_text(skill, include_category=False)
            candidate_normalized.update(extracted.keys())
        
        # Calculate matches
        matched = job_skills & candidate_normalized
        missing = job_skills - candidate_normalized
        extra = candidate_normalized - job_skills
        
        match_percentage = (len(matched) / len(job_skills) * 100) if job_skills else 0
        
        return {
            'match_percentage': round(match_percentage, 2),
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'additional_skills': list(extra),
            'total_required': len(job_skills),
            'total_matched': len(matched)
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating the technology extractor."""
    
    # Sample JSON data (the same as provided by user)
    sample_data = {
        "job_descriptions": [
            {
                "job_id": 1,
                "role": "Data Analyst",
                "location": "Hyderabad",
                "experience": "0-2 years",
                "description": "We are looking for a Data Analyst to collect, analyze, and interpret large datasets to help drive business decisions. The candidate will work with dashboards, perform data cleaning, and generate insights.",
                "skills": {
                    "programming": ["Python", "SQL"],
                    "libraries": ["Pandas", "NumPy", "Matplotlib"],
                    "databases": ["MySQL", "PostgreSQL", "MongoDB"],
                    "tools": ["Excel", "Power BI", "Tableau"],
                    "concepts": ["ETL", "Statistics", "Data Cleaning"]
                }
            },
            {
                "job_id": 2,
                "role": "Python Developer",
                "location": "Bangalore",
                "experience": "1-3 years",
                "description": "We are hiring a Python Developer to build backend services, automate workflows, and integrate APIs. The candidate should write clean and maintainable code.",
                "skills": {
                    "languages": ["Python"],
                    "frameworks": ["Django", "Flask", "FastAPI"],
                    "tools": ["Git", "Docker"],
                    "databases": ["SQLite", "PostgreSQL", "MongoDB"],
                    "cloud": ["AWS", "Azure"]
                }
            },
            {
                "job_id": 3,
                "role": "Machine Learning Engineer",
                "location": "Remote",
                "experience": "0-3 years",
                "description": "The ML Engineer will build ML pipelines, train models, deploy solutions, and optimize ML algorithms using large datasets.",
                "skills": {
                    "languages": ["Python"],
                    "ml_libraries": ["Scikit-Learn", "TensorFlow", "PyTorch", "Keras"],
                    "concepts": ["Deep Learning", "NLP", "Regression", "Classification"],
                    "tools": ["Jupyter Notebook", "Git"],
                    "cloud": ["AWS SageMaker", "Google Cloud AI"]
                }
            },
            {
                "job_id": 4,
                "role": "Full-Stack Developer",
                "location": "Chennai",
                "experience": "1-4 years",
                "description": "We are looking for a Full-Stack Developer to work on both frontend and backend applications, integrate services, and maintain high-performance systems.",
                "skills": {
                    "frontend": ["HTML", "CSS", "JavaScript", "React", "Angular"],
                    "backend": ["Node.js", "Express.js"],
                    "databases": ["MongoDB", "MySQL", "Firebase"],
                    "devops": ["Docker", "CI/CD"],
                    "cloud": ["AWS", "Azure"]
                }
            },
            {
                "job_id": 5,
                "role": "Cloud Engineer",
                "location": "Pune",
                "experience": "0-3 years",
                "description": "Cloud Engineer responsible for designing and maintaining cloud infrastructure and deploying scalable applications.",
                "skills": {
                    "cloud_platforms": ["AWS", "Azure", "Google Cloud"],
                    "services": ["EC2", "S3", "IAM", "Lambda", "VPC"],
                    "tools": ["Terraform", "Kubernetes", "Docker"],
                    "languages": ["Python", "Bash"],
                    "networking": ["Load Balancers", "Firewalls"]
                }
            }
        ]
    }
    
    print("=" * 70)
    print("JOB DESCRIPTION TECHNOLOGY EXTRACTOR")
    print("=" * 70)
    
    # Initialize extractor
    extractor = TechnologyExtractor()
    
    # Extract technologies from all job descriptions
    results = extractor.extract_from_json(sample_data)
    
    # Display results for each job
    print("\n📋 EXTRACTED TECHNOLOGIES BY JOB\n")
    print("-" * 70)
    
    for result in results:
        print(f"\n🔹 Job ID: {result['job_id']}")
        print(f"   Role: {result['role']}")
        print(f"   Location: {result['location']}")
        print(f"   Experience: {result['experience']}")
        print(f"\n   Technologies Found:")
        
        for category, techs in result['extracted_technologies'].items():
            tech_names = [t['name'] for t in techs]
            print(f"   • {category.replace('_', ' ').title()}: {', '.join(tech_names)}")
        
        print("-" * 70)
    
    # Generate and display summary
    summary = extractor.get_technology_summary(results)
    
    print("\n📊 SUMMARY STATISTICS\n")
    print(f"Total Jobs Analyzed: {summary['total_jobs_analyzed']}")
    print(f"Unique Technologies Found: {summary['unique_technologies_found']}")
    
    print("\n🏆 Top 10 Most In-Demand Technologies:")
    for i, (tech, count) in enumerate(summary['top_technologies'][:10], 1):
        print(f"   {i:2}. {tech}: {count} mentions")
    
    print("\n📁 Technologies by Category:")
    for category, techs in summary['technologies_by_category'].items():
        print(f"\n   {category.replace('_', ' ').title()}:")
        for tech, count in list(techs.items())[:5]:
            print(f"      - {tech}: {count}")
    
    print("\n👔 Technologies by Role:")
    for role, techs in summary['technologies_by_role'].items():
        print(f"\n   {role}:")
        for tech, count in list(techs.items())[:5]:
            print(f"      - {tech}: {count}")
    
    # Export results to JSON
    output_data = {
        'extraction_results': results,
        'summary': {
            'total_jobs_analyzed': summary['total_jobs_analyzed'],
            'unique_technologies_found': summary['unique_technologies_found'],
            'top_technologies': [{'name': t[0], 'count': t[1]} for t in summary['top_technologies']],
            'technologies_by_category': summary['technologies_by_category'],
            'technologies_by_role': summary['technologies_by_role']
        }
    }
    
    # Save to file
    output_file = 'extracted_technologies.json'
    JSONHandler.save_to_file(output_data, output_file)
    print(f"\n✅ Results saved to: {output_file}")
    
    return output_data


# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def extract_technologies_from_file(input_file: str, output_file: str = None) -> Dict:
    """
    Utility function to extract technologies from a JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Optional path to save results
    
    Returns:
        Dictionary with extraction results
    """
    # Load data
    json_data = JSONHandler.load_from_file(input_file)
    
    # Extract technologies
    extractor = TechnologyExtractor()
    results = extractor.extract_from_json(json_data)
    summary = extractor.get_technology_summary(results)
    
    output_data = {
        'extraction_results': results,
        'summary': summary
    }
    
    # Save if output file specified
    if output_file:
        JSONHandler.save_to_file(output_data, output_file)
    
    return output_data


def add_custom_technologies(custom_tech: Dict[str, Dict[str, List[str]]]) -> TechnologyExtractor:
    """
    Create extractor with custom technology definitions.
    
    Args:
        custom_tech: Dictionary of custom technologies to add
                    Format: {'category': {'TechName': ['variation1', 'variation2']}}
    
    Returns:
        TechnologyExtractor with extended database
    """
    extended_db = TECHNOLOGY_DATABASE.copy()
    
    for category, technologies in custom_tech.items():
        if category in extended_db:
            extended_db[category].update(technologies)
        else:
            extended_db[category] = technologies
    
    return TechnologyExtractor(extended_db)


if __name__ == "__main__":
    main()
