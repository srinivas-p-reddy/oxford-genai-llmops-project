# _Artificial Intelligence: Generative AI, Cloud and MLOps (online), University of Oxford_: LLMOps Project
## Introduction and Background
Oxford GenAI Course project. The aim of this project is to help students understand some of the key concerns and process flows required for AI engineering.

### What is AI engineering?
For the purposes of this course we use the term ‘AI engineering’ to refer to:
The discipline of building end-to-end applications that leverage Artificial Intelligence.
Note that this definition will also include within it the related concept of ‘Machine Learning Engineering’ (ML Engineering), which we consider a subset. These concepts will be explained in more detail in the rest of this document.

### What do AI engineers do?
Given the above description, AI engineering is necessarily concerned with AI systems in a relatively holistic sense and so it may prove difficult to succinctly define all activities that fall within its scope. The development of this field over the past 2-3 years, in particular, has however suggested a convergence on a certain set of activities that are core to the discipline:

* Software Engineering Best Practice
* Infrastructure & Cloud engineering.
* Data processing.
* Prompt Engineering.
* Orchestration.
* Model and System Performance Validation.
* Model and System Performance Monitoring.
* API Integration Engineering.
* System Design and Deployment Patterns.

Those who are aware of the disciplines of ML Engineering & Machine Learning Operations (MLOps) will see a lot of overlap in terms of these topics (with the exception of 3). This course module will help you build skills and knowledge in the area of AI engineering by building on a foundation of core MLOps practice and then extending these skills for the era of Foundation Models and Generative AI.

### Background Reading
As part of this course you have all been provided with an e-copy of [Machine Learning Engineering with Python, 2nd Edition](https://www.packtpub.com/product/machine-learning-engineering-with-python-2nd-edition/9781801078031) by [Andy McMahon](https://www.linkedin.com/in/andrew-p-mcmahon/). We will use this book to develop our core skills in AI engineering before augmenting it with new material.

The book is fundamentally a book about machine learning engineering, which can be thought of as a precursor to AI engineering. Many of the core foundational concepts from ML engineering carry over into AI engineering.

To augment your learning from the lectures in the course, the following topics are covered in the identified chapters in the book. I would suggest reading these and following some of the coding exercises within the book to prepare you for the  project (discussed in a later section):

1. Software Engineering Best Practice:
  * The ML and AI Development Lifecycle [Chapter 2.3]
  * Software Engineering & Python Fundamentals [Chapter 1.5, 4.1, 4.2 and 4.5]
  * CI/CD with GitHub Actions [Chapter 2.3 and 5.6]

2. Infrastructure & Cloud engineering:
  * AWS Services & AWS CLI [Throughout the book]
  * Kubernetes [Chapter 6.4 and 8.7]

3. Data Processing & Storage:
  * ETL Fundamentals [Chapter 6]
  * Apache Spark [Chapter 6.2]
  * Ray [Chapter 6.5]

4. Prompt Engineering:
  * Basic Langchain [Chapter 7.2]

5. Orchestration.
  * Apache Airflow [Chapter 5.6, 9.4 and 9.5]
  * Kubeflow [5.7]

6. Model and System Performance Validation & Monitoring [Chapter 3, Chapter 5, Chapters 8 & 9].

7. System Design and Deployment Patterns:
  * AI Application Patterns:
    * One-shot Q&A [Chapter 7.2]

This  project will build on some of the topics covered in the material above and provide a specific AI engineering bent to it. See details on the Project for more.

### AI Engineering vs ML Engineering

## Project
This project will complement the background study material with guided hands on work on the following topics:

1. CI/CD - you will learn how to use GitHub Actions in order to automate the testing and deployment of your code base. You will also learn how to use Makefiles to simplify complex application builds.
2. Docker - you will learn how to use containerisation to build microservices required for an AI use case.
3. AWS - you will learn what cloud services available in AWS are useful for AI applications and will get to know how to programmatically access them. This will include using the flagship AWS Bedrock service for foundation model serving.
4. LLM Traces - you will learn how to implement open source tools that allow you to monitor, track and debug interactions with LLMs in your application.
5. Advanced RAG- you will learn how to take a naive RAG implementation that does basic retrieval and generation and augment it with techniques such as query expansion. Importantly you will do this *without* using an off-the-shelf framework and will code everything natively in Python. This will give you insights into what goes on under the hood in the most popular libaries and frameworks.
6. Guardrails - We will explore how to apply Bedrock guardrails to protect against undesired outcomes in our AI application.
7. Databases - You will use Postgresql, one of the most popular production ready databases in the world, along with its `pgvector` extension to store and manipulate vectorised embeddings.
8. Ollama - We will use the Ollama solution in order to run LLMs locally for development and testing of your application, before moving to cloud hosted Foundation Models.

Other concepts which will be utilised in this project that are covered in the book in detail are:

9. Testing - You will see how to write advanced tests for async functions, for data intensive functions and for AI workflows.
10. Functional Programming (and more Python fundamentals) - This project is written in a functional style and will show you how to write code that is readable and maintainable even when not using Object Orientated Programming (OOP). The code will also show you how to write modular code that is easy to test, extend and adapt to multiple use cases.
11. Config driven development - We'll show you can leverage strong configuration principles in order to reduce duplication and make your code more maintainable.


### Structure
The repository has the following tree structure:

```
├── LICENSE
├── README.md
├── rag-app
│   ├── Makefile
│   ├── data
│   ├── deploy
│   │   ├── cloudformation
│   │   ├── docker
│   │   │   ├── llm-server
│   │   │   │   └── docker-compose.yml
│   │   │   └── postgres
│   │   │       ├── deduplicate.sql
│   │   │       ├── docker-compose.yaml
│   │   │       ├── init_pgvector.sql
│   │   │       └── pgvector.Dockerfile
│   │   └── scripts
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── server
│   │   ├── __init__.py
│   │   ├── config
│   │   │   ├── dev.yaml
│   │   │   └── prod.yaml
│   │   └── src
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── config_loader.py
│   │       ├── controllers
│   │       │   ├── __init__.py
│   │       │   ├── generation.py
│   │       │   ├── health_check.py
│   │       │   ├── rag_pipeline.py
│   │       │   └── retrieval.py
│   │       ├── database
│   │       │   ├── __init__.py
│   │       │   ├── db_session.py
│   │       │   └── vector.py
│   │       ├── ingestion
│   │       │   ├── __init__.py
│   │       │   ├── arxiv_client.py
│   │       │   ├── embeddings.py
│   │       │   ├── pipeline.py
│   │       │   └── utils.py
│   │       ├── main.py
│   │       ├── models
│   │       │   ├── __init__.py
│   │       │   ├── document.py
│   │       │   ├── generated_response.py
│   │       │   ├── query.py
│   │       │   └── user_interaction.py
│   │       ├── services
│   │       │   ├── __init__.py
│   │       │   ├── generation_service.py
│   │       │   └── retrieval_service.py
│   │       └── settings.py
│   └── tests
│       └── services
│           ├── __init__.py
│           ├── test_generation_service.py
│           └── test_retrieval_service.py
└── requirements.txt
```

This is based around the classic controller/service architecture, utilising FastAPI, PostgresQL and Ollama (for local LLM serving) and Amazon Bedrock (for remote LLM serving)

Here is a diagram outlining the main logical components of the application and how data flows through the app.:

```mermaid
flowchart LR
    %% Define Main Components in Horizontal Layout
    A[Client Query] --> B[Orchestrator]

    %% Service Components in Workflow
    B --> C[Query Expansion]
    C --> D[Retrieval Service]
    D --> E[Reranking Service]
    E --> F[Generation Service]
    F --> G[Generated Response to Client]

    %% Data Storage and Tracing
    D -->|Fetch Data| H[Knowledge Base]
    F --> I[Tracing & Monitoring]

    %% Styling (optional)
    classDef service fill:#f9f,stroke:#333,stroke-width:2px;
    classDef datastore fill:#bbf,stroke:#333,stroke-width:2px;
    class B,C,D,E,F,G service;
    class H,I datastore;
```
## Usage
The Makefile has been designed so that building and running the application and its components is as simple as possible.

To build the database:

```
make build-db
```

To download the data locally:

```
make download-data
```

To run the ingestion pipeline:

```
make run-ingestion
```

To run the FastAPI app:

```
make run-app
```

To run the tests

```
make test
```

To build the ollama service:

```
make build-ollama
```

To run the ollama service:

```
make run-ollama
```

Then to destroy the components, stop the app and then:

```
make remove-ollama
make remove-db
```

### AWS
1. First, sign up to get an AWS account if you don't already have one.
2. Next, I suggest configuring a budget alert (for $1 a day is fine).
3. Configure an IAM role in the AWS management console https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-role.html:
 1. Select 'AWS Account' for the trusted entity.
 2. Select your own account.
 3. For permissions policies, add permission for full access to Bedrock, ElasticContainerRegistry, ECS, EC2, CloudWatch (v2), CloudFormation, AmazonS3, AWSLambdaBasicExecutionRole.
 4. I've called this role Bedrock-Dev-FullAccess-Role

 ![alt text](assets/image-bedrock-role1.png)

 5. Create a user in the IAM centre, call it 'cli-access-user'


https://community.aws/content/2b6vVO87SMvy1cY70GeinjH5ZX3/multimodal?lang=en
https://github.com/suryakva/genai-titan-image-generator

#### Bedrock

**Objective**: Set up access to AWS Bedrock, enabling you to call its APIs through the AWS CLI.

**Step 1: Understand the Components**

1. IAM (Identity and Access Management): This is how AWS controls “who can do what.”
  * Users: Represent people (you or your students).
  * Roles: Represent permissions you temporarily “assume” to access AWS services.
  * Policies: Define “what actions” are allowed for users or roles.
2. Why Roles for Bedrock?
  * Roles are a secure way to give temporary access to AWS services (like Bedrock) without needing static credentials hardcoded anywhere.

**Step 2: Set Up AWS CLI Credentials Locally**

Before using AWS CLI, you need to authenticate with your AWS account:

1. Log in to the AWS Console:
  * Go to AWS Management Console.
  * Log in as the root user (or an IAM user with admin privileges).
2. Create an IAM User for Yourself (Best Practice):
  * Navigate to IAM > Users > Add Users.
  * Enter a username, e.g., AdminUser.
	* Check Programmatic Access (for CLI use).
	* Attach the policy AdministratorAccess (for simplicity during the course; tighten permissions later if desired).
	* Finish and download the Access Key ID and Secret Access Key.
3. Configure the AWS CLI:
  * Open your terminal and run:
  ```bash
  aws configure
  ```
  Enter the following when prompted:
  * AWS Access Key ID: Copy from the IAM user creation step.
  * AWS Secret Access Key: Copy from the IAM user creation step.
  * Default Region: Enter your preferred AWS region (e.g., us-east-1).
  * Output Format: json.
4. Test Your AWS CLI Configuration:
  * Run the following command:
  ```bash
  aws sts get-caller-identity
  ```
  You should see a response like this:
  ```
  {
  "User": {
    "Arn": "arn:aws:sts::123456789012:user/AdminUser",
    "UserId": "AIDEXAMPLEEXAMPLEEXAMPLE",
    "UserName": "AdminUser"
  },...
  }
  ```
  This indicates that your AWS CLI is configured correctly.

**Step 3: Create the role for Bedrock**

1. Navigate to IAM > Roles > Add Roles.
2. Enter a name for the role, e.g., Bedrock-Dev-FullAccess-Role.
3. Enter an AWS Managed Policy ARN, e.g., arn:aws:iam::aws:policy/AdministratorAccess.
4. Save the role.
5. Attach the policy BedrockFullAccess (for simplicity during the course; tighten permissions later if desired).
6. Copy the role ARN, e.g., arn:aws:iam::123456789012:role/Bedrock-Dev-FullAccess-Role.

**Step 4: Configure the AWS CLI to Assume the Role**
1. Create a Policy for Bedrock Access:
  * Go to IAM > Policies > Create Policy.
  * Choose the JSON tab and paste this policy:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "bedrock:InvokeModel",
          "bedrock:ListFoundationModels",
          "bedrock:ListCustomModels"
        ],
        "Resource": "*"
      }
    ]
  }
  ```
  * Click Next and name the policy, e.g., BedrockAccessPolicy.
  * Click Create.

2. 	Create a Role:
  * Go to IAM > Roles > Create Role.
  * Select AWS Service and choose EC2 (or any service; we’ll explain this later for simplicity).
  * Attach the BedrockAccessPolicy you created.
  * Name the role, e.g., Bedrock-Dev-Access-Role.

**Step 4: Assume the role for CLI use**
Now we are going to assume the role and get temporary credentials.

1. Run the following command to assume the newly created Bedrock-Dev-Access-Role (or substitute for whatever you have named it):
```bash
aws sts assume-role \
    --role-arn arn:aws:iam::<your-account-id>:role/Bedrock-Dev-Access-Role \
    --role-session-name CLI-Session
```
2. The output from the previous command will look something like:
```bash
{
    "Credentials": {
        "AccessKeyId": "AKIAIOSFODNN7EXAMPLE",
        "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "SessionToken": "AQoDYXdzEJzUvlyK...EXAMPLEKEY",
        "Expiration": "2023-07-11T19:00:00Z"
    }...
}
```
  These are temporary credentials that you can use to access Bedrock.
3. Use the returned credentials and export them as environment variables:
```bash
export AWS_ACCESS_KEY_ID=ASIA...
export AWS_SECRET_ACCESS_KEY=wJalr...
export AWS_SESSION_TOKEN=FQoGZXIvYXdz...
```

**Step 5: Confirm you can succesfully assume the role and access Bedrock**

Run the following command to list the available models:
```bash
aws bedrock list-foundation-models --region <your-region>
```
If this returns some json (it should be a rather large json) then we are done.

**Step 6: Enable model access in AWS Management Console**
Using the admin role for your account you should enable the models you want to use. At a minimum, you should enable the AWS Titan G1 - Express and Titan Text Embeddings V2 models for this RAG use case.

![alt text](assets/image-bedrock-model-enable1.png)

**Step 7 [Optional]: Set up secrets management for production**

If you want to set up secrets management for production, you can use AWS Secrets Manager to store your API key and other sensitive information. This can be done using the AWS CLI or in a script.
You can refer to the AWS documentation for more information on how to set up secrets management for production.

### Deployment


## Resources

#### General / RAG

1. https://blog.lancedb.com/guide-to-use-contextual-retrieval-and-prompt-caching-with-lancedb/
2. https://huggingface.co/learn/cookbook/agent_rag
3. https://docs.astral.sh/ruff/formatter/#format-suppression

#### Agentic RAG
1. https://github.com/cobusgreyling/LlamaIndex/blob/d8902482a247c76c7902ded143a875d5580f072a/Agentic_RAG_Multi_Document_Agents-v1.ipynb
# CI Test
# CI Test new
