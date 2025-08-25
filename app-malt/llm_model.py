import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
from solid_step_helper import clean_up_llm_output_func
import networkx as nx
import jsonlines
import json
import re
import time
import sys
import numpy as np
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain 
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent, ReAct_PromptAgent
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
import torch.multiprocessing as mp

# Load environ variables from .env, will not override existing environ variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)
# For Azure OpenAI GPT4
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatGoogleGenerativeAI
import getpass

mp.set_start_method('spawn', force=True)

prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """

class GoogleGeminiAgent:
    def __init__(self, prompt_type="base"):
        # Only ask for API key when choosing Gemini.
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_type = prompt_type
        
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()
        
    def call_agent(self, query):
        print("Calling Google Gemini with prompt type:", self.prompt_type)

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(query)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt.example_prompt)
            print("\nPrefix:", prompt.prefix)
            print("\nSuffix:", prompt.suffix)
        else:
            print(prompt.template.strip())
        print("-" * 80 + "\n")

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code


class AzureGPT4Agent:
    def __init__(self, prompt_type="base"):
        self.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    @staticmethod
    def configure_environment_variables():
        """Authenticates with Azure OpenAI and sets environment variables (prompting the user when necessary) if not already set."""
        # TODO: Is there a way to get the latest (stable) version programatically?
        DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-01"

        # Set the API_KEY to the token from the Azure credential
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            try:
                credential = DefaultAzureCredential()
                os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
            except Exception as e:
                print("Error retrieving Azure OpenAI API key (authenticating with Entra ID failed):", e)
                os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Please enter your Azure OpenAI API key: ")
        # Get the endpoint of deployed AzureGPT model.
        if "AZURE_OPENAI_ENDPOINT" not in os.environ:
            os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Please enter your Azure OpenAI endpoint: ")
        # Get the deployment name
        if "AZURE_OPENAI_DEPLOYMENT_NAME" not in os.environ:
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = getpass.getpass("Please enter your Azure OpenAI deployment name: ")
        # Get the OpenAI API version
        if "AZURE_OPENAI_API_VERSION" not in os.environ:
            os.environ["AZURE_OPENAI_API_VERSION"] = DEFAULT_AZURE_OPENAI_API_VERSION

    def call_agent(self, query):
        print("Calling GPT-4o with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(query)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt.example_prompt)
            print("\nPrefix:", prompt.prefix)
            print("\nSuffix:", prompt.suffix)
        else:
            print(prompt.template.strip())
        print("-" * 80 + "\n")

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
   
class QwenModel:
    def __init__(self, prompt_type="base"):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq"
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    def call_agent(self, query):
        print("Calling Qwen with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(query)
            # For few-shot semantic, we need to format with the specific query
            prompt_text = prompt_template.format(input=query)
        elif self.prompt_type == "few_shot_basic":
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            # For few-shot basic, format with the query
            prompt_text = prompt_template.format(input=query)
        else:
            # For base/cot prompts
            prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )
            prompt_text = self.prompt_agent.prompt_prefix + prompt_suffix
            prompt_text = prompt_text.format(input=query)

        # Print prompt template (keeping the same debugging output)
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt_template, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt_template.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt_template.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt_template.example_prompt)
            print("\nPrefix:", prompt_template.prefix)
            print("\nSuffix:", prompt_template.suffix)
        else:
            print(prompt_text.strip())
        print("-" * 80 + "\n")

        # Use vLLM's native interface for generation
        outputs = self.llm.generate(prompt_text, self.sampling_params)
        answer = outputs[0].outputs[0].text
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code

class QwenModel_finetuned:
    def __init__(self, prompt_type="base", model_path=None):
        self.model_name = "Fine-tuned-Qwen-7B"
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        # Set padding token to be same as EOS token, but with explicit attention mask handling
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            fp16=True
        ).eval()
        
        # Use the default generation config from the model
        self.llm.generation_config = self.llm.generation_config
        
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    def call_agent(self, query):
        print("Calling Fine-tuned Qwen with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(query)
            # For few-shot semantic, we need to format with the specific query
            prompt_text = prompt_template.format(input=query)
        elif self.prompt_type == "few_shot_basic":
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            # For few-shot basic, format with the query
            prompt_text = prompt_template.format(input=query)
        else:
            # For base/cot prompts
            prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )
            prompt_text = self.prompt_agent.prompt_prefix + prompt_suffix
            prompt_text = prompt_text.format(input=query)

        # Print prompt template (keeping the same debugging output)
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt_template, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt_template.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt_template.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt_template.example_prompt)
            print("\nPrefix:", prompt_template.prefix)
            print("\nSuffix:", prompt_template.suffix)
        else:
            print(prompt_text.strip())
        print("-" * 80 + "\n")

        # Explicitly create the attention mask to handle the warning
        # Use the modelscope chat method with attention mask handling
        try:
            # First try the existing method which might work despite the warning
            answer, _ = self.llm.chat(self.tokenizer, prompt_text, history=None)
        except Exception as e:
            print(f"Standard chat method failed with: {e}. Using fallback method with explicit attention mask.")
            # Fallback to manual generation with explicit attention mask
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.llm.device)
            # Generate with explicit attention mask
            with torch.no_grad():
                outputs = self.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=512,
                    temperature=0.0
                )
            # Decode the output skipping the input tokens
            answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
 

# ReAct agent
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
class ReAct_Agent:
    def __init__(self, prompt_type="react"):
        AzureGPT4Agent.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        self.prompt_agent = ReAct_PromptAgent()
        

    def call_agent(self, query):
        print("Calling ReAct agent withGPT-4o with prompt type:", self.prompt_type)
        
        prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        print(prompt_template.template.strip())
        print("-" * 80 + "\n")

        # Set up the Python REPL tool
        python_repl = PythonAstREPLTool()
        python_repl_tool = Tool(
            name = 'Python REPL',
            func = python_repl.run,
            description = '''
            A Python shell. Use this to execute python commands. 
            Input should be a valid python command. 
            When using this tool, sometimes output is abbreviated - make sure 
            it does not look abbreviated before using it in your answer.
            '''
        )
        print("Python REPL tool set up")

        # Set up the DuckDuckGo Search tool
        search = DuckDuckGoSearchRun()
        duckduckgo_tool = Tool(
            name = 'DuckDuckGo Search',
            func = search.run,
            description = '''
            A wrapper around DuckDuckGo Search. 
            Useful for when you need to answer questions about current events. 
            Input should be a search query.
            '''
        )
        print("DuckDuckGo Search tool set up")

        # Create an array that contains all the tools used by the agent
        tools = [python_repl_tool, duckduckgo_tool]
        print("The ReAct agent can access the following tools: Python REPL, DuckDuckGo Search")

        # Create a ReAct agent
        react_format_prompt = hub.pull('hwchase17/react')
        agent = create_react_agent(self.llm, tools, react_format_prompt)
        print("ReAct agent created")

        print("Setting up agent executor...")
        agent_executor = AgentExecutor(
            agent=agent, 
            tools = tools,
            verbose = True, # explain all reasoning steps
            handle_parsing_errors=True, # continue on error 
            max_iterations = 2 # try up to 3 times to find the best answer
        )
        print("ReAct agent executor set up")
        # Format the input correctly based on the prompt type

        output = agent_executor.invoke({'input': prompt_template.format(input=query)})
        # Return the output in the same format as other agents
        answer = output['output']
        print("ReActanswer: ", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code


class OpenRouterAgent:
    def __init__(self, prompt_type="base", model_name="anthropic/claude-3.5-sonnet"):
        self.model_name = model_name
        
        # Configure API key
        if "OPENROUTER_API_KEY" not in os.environ:
            os.environ["OPENROUTER_API_KEY"] = getpass.getpass("Enter your OpenRouter API key: ")
        
        # Initialize OpenAI client with OpenRouter endpoint
        import openai
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        
        self.prompt_type = prompt_type
        
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()
    
    def call_agent(self, query):
        print(f"Calling OpenRouter ({self.model_name}) with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(query)
            prompt_text = prompt_template.format(input=query)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            prompt_text = prompt_template.format(input=query)
        else:
            # For base/cot prompts
            prompt_template = None
            prompt_text = self.prompt_agent.prompt_prefix + prompt_suffix
            prompt_text = prompt_text.format(input=query)

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt_template, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt_template.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt_template.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt_template.example_prompt)
            print("\nPrefix:", prompt_template.prefix)
            print("\nSuffix:", prompt_template.suffix)
        else:
            print(prompt_text.strip())
        print("-" * 80 + "\n")

        try:
            # Call OpenRouter API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            
            answer = response.choices[0].message.content
            print("model returned")
            code = clean_up_llm_output_func(answer)
            return code
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return ""
