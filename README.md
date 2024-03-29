
# [Decay No More: A Persistent Twitter Dataset for Learning Social Meaning](https://arxiv.org/pdf/2204.04611.pdf)
#### Best Paper Award by 1st Workshop on Novel Evaluation Approaches for Text Classification Systems on Social Media @ ICWSM-2022
![](https://github.com/chiyuzhang94/PTSM/blob/main/pic/title.png)
## Datasets
To download PTSM data, please fill out this [registration form](https://forms.gle/ug7Ti1W9DCrJHYn26). We will share the dataset after the registration. 
![](https://github.com/chiyuzhang94/PTSM/blob/main/pic/dataset.png)
## Paraphrase Model
We fine-tune T5-base on four paraphrase gold datasets as shown below.
| Dataset     | Domain         | \# of samples |
| ----------- | -------------- | ------------- |
| PIT-2015    | Tweet          | 3,789          |
| LanguageNet | Tweet          | 12,988         |
| Opusparcus  | Video subtitle | 462,846        |
| QQP         | Quora          | 149,263        |
| Total       | \-             | 625,097        |

You can find our model on: https://huggingface.co/UBC-NLP/ptsm_t5_paraphraser 
You can load this model by:
```python
from transformers import T5ForConditionalGeneration,T5TokenizerFast

model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
tokenizer = T5TokenizerFast.from_pretrained(model_name_or_path)
```

A prefix "paraphrase: " should be added in font of the input sequence, i.e.:
```python
input_st = "paraphrase: " + text + " </s>"
```

## Paraphrase Examples
| **Original Tweet** | **Paraphrase** | **Gold Label** |
| ----  | ------ | ------- | 
|USER but what I am doing is in my control, #AvoidMMT , you guys are #terrible | 1. You guys are horrible, avoid MMT <br /> 2. what I am doing is in my control, #AvoidMMT, you guys are terrifying <br /> 3. You guys are #terrorist. I have used everything I have to do. <br /> 4. You guys are awful, but I am going to stop doing it.| anger | 
|I hate when people say 'I need to talk to you or we need to talk.' My anxiety immediately goes up... | 1. I hate when people say 'I need to talk to you or we need to talk to you'. I guess that's the problem. <br /> 2. I hate when people tell me 'I need to talk to you or we need to speak' my anxiety immediately goes up. <br /> 3. Why am I afraid when people say 'I need to talk to you or we need to talk?' | anger | 
|This is the 47th wedding I've ruined. When I hit 50 I can retire. | 1. The 46th wedding I've ruined. When I hit 50 I can retire. It's nice to see yo <br /> 2. Here's the 47th wedding I've ruined. If I'm old enough to go on the 40s I can get married. <br /> 3. After a single wedding, I drew 47 weddings, and before I hit 50 I can retire" | humor | 
|Sorry guys I have absolutely no idea what time i'll be on cam tomorrow but will keep you posted. | 1. Sorry to disturb you. I have absolutely no idea what time I'll be on cam tomorrow. <br /> 2. I have absolutely no idea what time I'll be on camera tomorrow but I'll keep you posted | sadness | 
| Who wants to go with me to get my tattoo tomorrow? I'll buy you Dunkin doughnuts | 1. "I'll buy you Dunkin' Donuts for $5. <br /> 2. Who wants to go with me for my tattoo tomorrow? I'll buy you a Dunkin' Donuts. <br /> 3. Who wants to go with me to get my tattoo tomorrow? | neutral | 
| USER May I suggest, that you have a meal that is made with beans, onions &amp; garlic, the day before class. | 1. The day before class please eat beans, onions and garlic. Also see the videos <br /> 2. "The Day Before Class. You should make that meal, (do you think). <br /> 3. If you can eat just the day before class, make a wonderful meal with garlic, onions and beans. | joy | 

Please find more details in our paper:
```
@inproceedings{zhang2022decay,
   title={Decay No More: A Persistent Twitter Dataset for Learning Social Meaning},
   author={Zhang, Chiyu and Abdul-Mageed, Muhammad and Nagoudi, El Moatez Billah},
   booktitle ={Proceedings of 1st Workshop on Novel Evaluation Approaches for Text Classification Systems on Social Media (NEATCLasS)}, 
   year={2022},
   url = {https://arxiv.org/pdf/2204.04611.pdf},
   publisher = {{AAAI} Press}, 
}
```
