import time
import json
import os
import warnings
import re
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# Login Hugging Face
login(token=huggingface_token)
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, ReAct_PromptAgent
from datetime import datetime
from vllm import LLM, SamplingParams
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# For Azure OpenAI GPT4
from azure.identity import DefaultAzureCredential
from langchain.chat_models import AzureChatOpenAI
import getpass
# ReAct agent
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool

class LLMModel:
    """
    A simplified class for handling language models.

    Parameters:
    -----------
    model : str
        The name of the model to be used.
    max_new_tokens : int, optional
        The maximum number of new tokens to be generated (default is 20).
    temperature : float, optional
        The temperature for text generation (default is 0).
    device : str, optional
        The device to be used for inference (default is "cuda").
    api_key : str or None, optional
        The API key for API-based models (default is None).
    """

    @staticmethod
    def model_list():
        return [
            "Qwen/Qwen2.5-72B-Instruct",
            "GPT-Agent",
            "ReAct_Agent",
            "OpenRouter"
        ]

    def __init__(self, model: str, max_new_tokens: int = 256, temperature: float = 0.1, device: str = "cuda", api_key: str = None, vllm: bool = True, prompt_type: str = "cot", num_gpus: int = 1):
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.api_key = api_key
        self.vllm = vllm
        self.prompt_type=prompt_type
        self.num_gpus = num_gpus
        self.model = self._create_model()

    @staticmethod
    def extract_value(text, keyword):
        """Extract a specific value from the text based on a keyword."""
        # Format: "keyword": "value" (case insensitive)
        pattern = rf'"{keyword}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def extract_number_before_percentage(text):
        """Extract the number that appears before the '%' symbol in the text."""
        import re
        pattern = r'(\d+)(?=\s*%)'  # Looks for digits before the '%'
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None
    
    def _create_model(self):
        """Creates and returns the appropriate model based on the model name."""
        if self.model_name == "Qwen/Qwen2.5-72B-Instruct":
            return self._initialize_qwen()
        elif self.model_name == "GPT-Agent":
            return self._initialize_gpt_agent()
        elif self.model_name == "ReAct_Agent":
            return self._initialize_react()
        elif self.model_name == "OpenRouter":
            return self._initialize_openrouter()
        elif self.model_name == "YourModel":
            return self._initialize_YourModel()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported!")
        
    def _initialize_react(self):
        return ReAct_Agent(prompt_type=self.prompt_type)

    def _initialize_openrouter(self):
        """Initialize the OpenRouter model."""
        from openrouter_model import OpenRouterModel
        return OpenRouterModel(prompt_type=self.prompt_type)

    def _initialize_qwen(self):
        """Initialize the Qwen model."""
        if self.vllm:
            return Qwen_vllm_Model(
                model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                device=self.device,
                num_gpus=self.num_gpus
            )
        else:
            return QwenModel(
                model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                device=self.device,
                prompt_type=self.prompt_type
            )
    
    def _initialize_gpt_agent(self):
        """Initialize the GPT Agent model."""
        return GPTAgentModel(prompt_type=self.prompt_type)

    def _initialize_YourModel(self):
        """Initialize your model."""
        # ====== TODO: Specify parameters and return your own model instance here ======
        # Example: return YourModel(prompt_type=self.prompt_type)
        # ====== END TODO ======
        return YourModel

    def __call__(self, input_text: str, **kwargs):
        """Perform inference with the loaded model."""
        # Replace with actual inference logic
        return f"Generating response for: '{input_text}' using {self.model_name}"

class QwenModel:
    """
    A specialized class for handling Qwen/Qwen2.5-72B-Instruct models.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device, prompt_type="base"):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()
        self.prompt_type = prompt_type
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()

    def _load_model(self):
        """Load the Qwen model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        # Create BitsAndBytesConfig for 4-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device
        )

        # Load the Qwen model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device,
            quantization_config=quantization_config  # Use the quantization config
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        start_time = time.time()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            **kwargs
        )
        content = str(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        print(f'\n**BEGIN (RAW) LLM OUTPUT**\n{"=" * 50}\n{content}\n{"=" * 50}\n**END (RAW) LLM OUTPUT**\n')
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands

class Qwen_vllm_Model:
    """
    A specialized class for handling Qwen/Qwen2.5-72B-Instruct models with GPTQ 4-bit quantization.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device="cuda", prompt_type="base", num_gpus=1):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.num_gpus = num_gpus
        self._load_model()
        self.prompt_type = prompt_type
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()

    def _load_model(self):
        """Load the Qwen model using vllm with GPTQ 4-bit quantization."""

        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq",  # Enable GPTQ 4-bit loading
            gpu_memory_utilization=0.95,
            tensor_parallel_size=self.num_gpus,
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)

        start_time = time.time()

        # Generate response using vllm
        result = self.llm.generate([prompt], sampling_params=self.sampling_params)
        content = result[0].outputs[0].text
        print(f'\n**BEGIN (RAW) LLM OUTPUT**\n{"=" * 50}\n{content}\n{"=" * 50}\n**END (RAW) LLM OUTPUT**\n')

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands


class GPTAgentModel:
    """
    A specialized class for handling GPT Agent models.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    api_key : str
        The API key for GPT Agent.
    """

    def __init__(self, prompt_type="base"):
        self.prompt_type = prompt_type
        self._load_model()

    def _load_model(self):
        """Initialize the GPT Agent client."""
        self.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        if self.prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif self.prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        print("======GPT-4o successfully loaded=======")

    @staticmethod
    def configure_environment_variables(self):
        """Authenticates with Azure OpenAI and sets environment variables if not already set."""
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
            os.environ["AZURE_OPENAI_API_VERSION"] = self.DEFAULT_AZURE_OPENAI_API_VERSION

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()
        connectivitity_status = file_content + log_content

        # content = response.choices[0].message.content
        max_length = 127000  
        if len(connectivitity_status) > max_length:
            connectivity_status = connectivitity_status[:max_length]

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            input_data = {"input": connectivity_status}
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            input_data = {"input": connectivity_status}
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivity_status}
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivity_status}
        start_time = time.time()
        chain = LLMChain(llm=self.client, prompt=prompt)
        content = chain.run(input_data)
        print(f'\n**BEGIN (RAW) LLM OUTPUT**\n{"=" * 50}\n{content}\n{"=" * 50}\n**END (RAW) LLM OUTPUT**\n')
        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands

class ReAct_Agent:
    def __init__(self, prompt_type="react"):
        GPTAgentModel.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        self.prompt_agent = ReAct_PromptAgent()
        
    def predict(self, log_content, file_path, json_path, **kwargs):
        with open(file_path, 'r') as f:
            file_content = f.read()
        query = file_content + log_content
        print("Calling ReAct agent withGPT-4o with prompt type:", self.prompt_type)
        
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
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
            max_iterations = 2, # try up to 3 times to find the best answer
            return_intermediate_steps=True
        )
        
        print("ReAct agent executor set up")
        # Format the input correctly based on the prompt type

        start_time = time.time()

        # Invoke the agent executor with the input query
        output = agent_executor.invoke({'input': prompt_template.format(input=query)})
        end_time = time.time()

        intermediate_steps = output['intermediate_steps']  # Get the intermediate steps

        # Access the first tuple in the list
        if intermediate_steps:  
            print("Intermediate steps are not empty.")
            print("Intermediate steps:", intermediate_steps)
            print("end of intermediate steps")
            first_step = intermediate_steps[0]
            print("First step:", first_step)
            
            # Extract the AgentAction object from the tuple
            agent_action = first_step[0]
            
            # Retrieve the tool_input value
            tool_input = agent_action.tool_input
            print("ReAct agent output:", tool_input)
            
            machine = LLMModel.extract_value(tool_input, "machine")
            commands = LLMModel.extract_value(tool_input, "command")
        else:
            print("Intermediate steps is empty.")
            machine = ""
            commands = ""

        # Extract loss rate
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        # Print results
        print(f"Machine: {machine}")
        print(f"Commands: {commands}")
        print(f"Loss Rate: {loss_rate}")

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")
            
        with open(json_path, "r") as f:
            data = json.load(f)
        data.append({"packet_loss": loss_rate, "elapsed_time": end_time - start_time})
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        return machine, commands

class YourModel:
    """
    A specialized class for handling YourModel.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, prompt_type="base"):
        self.prompt_type = prompt_type
        self._load_model()

    def _load_model(self):
        """Load the your model and tokenizer."""
        # ====== TODO: Load your own model and tokenizer here ======
        # Example: self.llm = AutoTokenizer.from_pretrained(...)
        # ====== END TODO ======
        # Choose prompt agent based on the prompt type
        if self.prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif self.prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)

        start_time = time.time()

        # ====== TODO: Replace the following line with your actual inference logic ======
        # Example: content = self.llm.generate(prompt)
        content = ""
        # ====== END TODO ======

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands
