# Assignment 30 -Capstone - Multimodal Training

# Objective
-  To Create multimodal LLM which takes Text, Image and Audio and generates output as text

# Description
-   Phi-2 model is used as foundation model for training the multimodel llm. 
-   After pre training, the model is further fine tunned with [QLORA](https://arxiv.org/abs/2305.14314)
   
# Dataset

-   The dataset *LLaVA dataset* is used, which contains image files paired with conversation data. 
-   Each entry has a unique identifier (id), an image file (image), and a conversation history (conversations). 
-   These conversations simulate interactions between a human and an AI model, often asking questions about the content of the images.

## Step 1:Data-Processing   
### 01_preprocessing_dataset.ipynb:
- This notebook executes preprocessing the LLaVA dataset, for multimodal task. We focus on combining text and visual data as input
  
    - #### Fetching Data:
    The notebook provides a code block for fetching the dataset from Hugging Face's repository.

    - #### Data Preparation:
    The next step involves preparing the dataset for training. This includes loading the JSON file, extracting relevant for use in machine learning tasks.
  
    As part of the data preparation process, the conversations are extracted, and the text is cleaned and organized for the model’s input. Images are preprocessed to a standard format for easier use in neural network models.

## Step 2: Pre-training(Projection Model)  
### S30_pretraining.ipynb :
-   In this notebook pretraining models using PyTorch and Hugging Face's transformers library. Phi-2 model is used  process text input.The image inputs are coverted into embeddings which Phi-2 can understand using projection layer which generates projection model. Images are processed using  CLIP as base model (openai/clip-vit-base-patch32) to generate embeddings.In order to train project model, the data is passed through the projection layer. Clip model and Phi-2 models are frozen and projection model is only trained. These image embeddings are passed through a projection layer though ResBlock projection model, Resblock model ised to capture the context of the image.


    -   Teacher Forcing with Simulated Annealing:
    The notebook implements a teacher forcing simulated annealing scheduler. This method is used to gradually adjust the teacher forcing ratio over the training process:
    -   Teacher Forcing: This is a training technique where the model uses the ground truth output (instead of its own prediction) as input for the next time step.
    Simulated Annealing: The ratio of teacher forcing is scheduled to increase/decrease cyclically over iterations. This gradual change helps balance learning stability and the model’s ability to generalize.

    -   Model Setup:
    Models: The notebook defines models such as phi-2 (microsoft/phi-2) and OpenAI’s CLIP (clip-vit-base-patch32). These are pretrained models, which combine vision and language understanding.
    The image inputs are coverted into embeddings which Phi-2 can understand using projection layer which generates projection model. Images are processed using  CLIP as base model (openai/clip-vit-base-patch32) to generate embeddings.In order to train project model, the data is passed through the projection layer. Clip model and Phi-2 models are frozen and projection model is only trained. These image embeddings are passed through a projection layer though ResBlock projection model, Resblock model ised to capture the context of the image.

    ```python
    class SimpleResBlock(nn.Module):
        def __init__(self, phi_embed):
            super().__init__()
            self.pre_norm = nn.LayerNorm(phi_embed)
            self.proj = nn.Sequential(
                nn.Linear(phi_embed, phi_embed),
                nn.GELU(),
                nn.Linear(phi_embed, phi_embed)
            )
        def forward(self, x):
            x = self.pre_norm(x)
            return x + self.proj(x)
    ```
    -   The projection model output is augmented with an "end of image" token (IMAGE_TOKEN_ID = 23893) and passed to the Phi-2 model's forward method.The model is trained and captions are generated for all images. The loss is calulated by referring the ground truth captions and the predicted captions.

    -   Random Split and Data Handling: The notebook makes use of PyTorch’s DataLoader for managing data pipelines and includes dataset splitting logic (random_split), potentially indicating training-validation splitting.

    -   Focus on Efficient Training:
    The inclusion of optimizations like bitsandbytes and flash attention suggests a focus on memory-efficient training, particularly useful for training large models on GPUs with limited memory.

    -   The scheduler function for teacher forcing is designed to enable cyclic variation in the forcing ratio. This could be used to prevent overfitting while improving model generalization, making the training process adaptive and flexible.

   
-   Reference : [LLAVA Paper](https://arxiv.org/pdf/2304.08485.pdf)

-   Resources: 
    -   GPU Used: NVIDIA-A100 40GB GPU (Dedicated). 
    -   Training duration 5.5 hours with 2 epochs  and iteration 20000 per epoch 
    -   Loss strats from 7.2 and declines to 5.6

-   Training Logs:
    Refer S30--pretrain_output.log


### Stage 2: Fine-training S29_finetunning.ipynb :


-   The notebook contains training a multi-modal model, likely for a  Visual QA, where both images and questions are input to the model, and the model generates a relevant answer. This involves the combination of pre-trained models (CLIP for image understanding and phi-2 for text understanding), fine-tuning them on a specific dataset, and tracking the model's performance using tools like Weights & Biases.

-   Llava Instruct 150 k dataset is used to fine tune LLM model by understaning the dialouges from image and model. QLora techinque is used for fine tunning and genates adapters(model parameters and weights).The training iniitiated  with starting loss of 6.6 which subsequently decreased to 3.4 over 3 epochs and  100000 steps per epoch.

- Detailed:
-   CLIP Model: A model developed by OpenAI that allows images and text to be encoded into a shared embedding space, enabling cross-modal tasks like image captioning or visual question answering (VQA).
-   Parameter-Efficient Fine-Tuning (PEFT): A technique to fine-tune large models by only updating a small subset of parameters, making the process faster and more memory efficient.
-   Multi-Modal Model: Combines different types of data (text and images in this case) to make predictions or generate outputs.

-   Resources: 
    -   GPU Used: NVIDIA-A100 40GB GPU (Dedicated). 
    -   Training duration 12 hours with epoch 3 and iteration 100000 per epoch 
    -   Loss strats from 6.6 and declines to 3.4(at the 67% of epoch 3)


### Huggingface Gradio App:
-    The app.py script is a multimodal AI application that integrates image, audio, and text inputs using pre-trained models like CLIP (for vision tasks), Phi-2 (for text generation), and WhisperX (for audio transcription). The script sets up tokenizers and processors for handling inputs and defines a custom residual block (SimpleResBlock) to transform embeddings for more stable learning. 
  
-    After loading pretrained and fine-tuned weights for both the projection and residual layers, it implements the model_generate_ans function, which processes inputs from different modalities, combines their embeddings, and generates responses sequentially.
-     This model handles tasks like image embedding extraction, audio transcription and embedding, and text tokenization to predict responses. 
-     The app features a Gradio interface where users can upload images, record or upload audio, and submit text queries, receiving multimodal answers through a web interface. 

https://huggingface.co/spaces/Vvaann/Capstone



### Improvements:
-   Hyper parameter tuning
-   Using Phi3 inplace of phi2 for increasing overall perforamce 
