# NLP Conversational Agent

This project conducts conversational agent/chatbot task using GPT-2 for text generation. Three versions were developed, one a zero-model, a model fine-tuned on the dialogue language only, and a model fine-tuned on the dialogue and accompany structured data.

## Dataset

MultiWOZ 2.2 dataset (for detailed descriptions see https://arxiv.org/pdf/1810.00278.pdf and https://aclanthology.org/2020.nlp4convai‐1.13.pdf). This dataset contains multi-turn conversations between an agent and a user trying to accomplish a task in multiple domains (e.g., find a restaurant, book a hotel, etc.). Most turns in the conversation are annotated with specific values related to actions, user-intent, descriptive slot/value pairs and state-tracking, among others.

## Task

Improve the baseline for the the model that generates both sides of the dialogue by using decoding and different evaluation metrics.

## Method
Decoding methods used:
1. Top-K sampling:
    Top K sampling ensures that only top k probable tokens must be considered for a generation. Where top_k is the number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity and the default value is set to 50. This way, Top-K provides a solution for Top-p which picks the highly suggested words and devolves into repeated words. Top-K’s approach also allows other high scoring tokens a chance to be selected and improves the quality of text generation
2. Temperature Sampling:
    Temperature simply adjusts the probability distribution of the words. We can say that at lower temperatures our model is more deterministic and at higher temperature values, it’s less so. To find what works best for us, we altered between different parameter values for temperature and finally set the temperature to 0.7.

Evaluation metrics for monitoring and improving the performance of the model:
1. ROUGE
2. BLUE
3. BERT_SCORE 


## Results

| Metric | Fine-tuning 1 | Fine-tuning 2 | Improvement|
| :------------ |:---------------:| -----:|-----:|
| BLEU| 0.104 | 0.338 | 225%|  
| ROUGE| 0.244 | 0.718 | 194% |
| BERTscore| 0.866 | 0.916 | 6% |

**BLEU:** For Bleu we see an improvement of more than double, since it is notorious that the pred text now has a greater number of n-grams identical to the ref. Also note that both pred texts have a length greater than the ref, which avoids the brevity penalty. But, the change of order of some words works against it in the score.

**ROUGE:** The improvement in Rouge is very similar to Bleu following a very similar trend as we have seen in the total average values. However, we can see that the absolute value of Rouge is much larger than Bleu, this helps ROUGE to have a greater range of visibility when evaluating the generation of text, since we could have a set of sentences where BLEU is usually close to 0, but in those same cases ROUGE does take reasonable values, even if they are low, it is possible to compare some generations with others, giving this last metric greater sensitivity in the measurement.

**BERTscore:** The Bertscore improvement is the lowest of all, with only 6%. This is because initially, the value is already very high. This metric mainly measures fluency, so when even fine-tuning 1 uses different words in pred than ref, the sentence makes sense, which explains the initial high value. The disadvantage of this metric is that the generated text may not mean exactly the same as the reference, as we can see in the example that is indicating a different address.

### Examples

Beam search, fine-tuning 1:
ref:   The address is Cambridge Leisure Park Clifton Way Cherry Hinton. The phone number is 01223 412430. [END]
pred:  wok is located at 191 histon road chesterton and its phone number is 01223 362456. Is there anything else I can help you with?

Beam search, fine-tuning 2:
​​ref:   The address is Cambridge Leisure Park Clifton Way Cherry Hinton. The phone number is 01223 412430. [END]
pred:  address for Pizza Hut Cherry Hinton is G4 Cambridge Leisure Park Clifton Way Cherry Hinton and the phone number is 01223 323737.

## Conclusion

**Strengths of different metrics:**
1. Sacrebleu as a metric for benchmarking rather than bleu. One weakness of bleu is that it expects the text to be tokenized. Sacrebleu handles this efficiently through the internal tokenization step. Hence, it is often used as a metric for benchmark.
Handling zero-counts in n-grams
2. Bleu score tends to become zero when there are missing n-grams. We can make the metric smoother by adding a constant to the numerator. From the results, we observe that in most cases after adding a smoothing technique the bleu score is slightly better. 

**Drawbacks of different metrics**
1. Handling Similar words or synonyms with bleu score. One of the major drawbacks of the bleu score is that it fails to take synonyms or similar words into account and has fragile heuristics. Although the two statements are identical from the human perspective, they are not given a bleu score of 100.

2. Assigning the brevity penalty in bleu score. Although brevity penalty works well in other text generation tasks like language translation. It might not work well with chatbots as we mostly convey messages with short texts.


