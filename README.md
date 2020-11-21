# AraBert Extractive Summarizer

This repo is forked from Bert Extractive Summarizer repo. This tool utilizes the HuggingFace Pytorch transformers library
to run extractive summarizations. This works by first embedding the sentences, then running a clustering algorithm, finding 
the sentences that are closest to the cluster's centroids.

Paper: http://dstore.alazhar.edu.ps/xmlui/handle/123456789/633

## Install

```bash
 pip3 install -r requirements.txt
```

## How to Use

#### Simple Example
```python
from summarizer import Summarizer
from transformers import AutoTokenizer, AutoModel
from rouge import Rouge

body='''
أعلن اليوم الخميس في ماليزيا عن دخول ملك البلاد في حجر صحي بعد إصابة 7 عاملين في القصر بفيروس كورونا، ليكون بذلك أحدث زعماء العالم التحاقا بقائمة القادة الذين تحوم حولهم شبهة الإصابة بهذا الفيروس.
وقال مشرف القصر الوطني في ماليزيا أحمد فاضل شمس الدين اليوم إن الملك السلطان عبد الله رعاية الدين المصطفى بالله شاه والملكة الحاجة عزيزة أمينة ميمونة الإسكندرية قد خضعا لفحص طبي واختبار تشخيصي للفيروس، حيث جاءت نتائج تحاليلهما سلبية.
وقال إن الملك والملكة يخضعان حاليا للحجر الصحي في القصر، ولن يقبلا أي زيارة أو مقابلة رسمية إلى أن تنتهي فترة الحجر الصحي التي بدأت أمس ومن المقرر أن تستمر لمدة 14 يوما.
ويوم أمس الأربعاء، أعلن مقر إقامة ولي العهد البريطاني الأمير تشارلز إصابة الأمير بفيروس كورونا.
وقال متحدث باسم مقر إقامة الأمير تشارلز ثبتت إصابة الأمير تشارلز بفيروس كورونا، لقد ظهرت عليه أعراض طفيفة لكن صحته جيدة، وكان يعمل من البيت طوال الأيام الماضية كالمعتاد.
'''
albert_model = AutoModel.from_pretrained('asafaya/bert-base-arabic')
albert_tokenizer = AutoTokenizer.from_pretrained('asafaya/bert-base-arabic')

modelSummarizer = Summarizer(custom_model=albert_model, custom_tokenizer=albert_tokenizer)
result = modelSummarizer(body)
generated_summary = ''.join(result)
print(generated_summary)

```

### Evaluation Using ROUGE

```python

summary_evaluation = '''
أعلن اليوم الخميس في ماليزيا عن دخول ملك البلاد في حجر صحي بعد إصابة 7 عاملين في القصر بفيروس كورونا، ليكون بذلك أحدث زعماء العالم التحاقا بقائمة القادة الذين تحوم حولهم شبهة الإصابة بهذا الفيروس.
ويوم أمس الأربعاء، أعلن مقر إقامة ولي العهد البريطاني الأمير تشارلز إصابة الأمير بفيروس كورونا.
ومساء الأحد الماضي، أعلن في ألمانيا عن الاشتباه في إصابة المستشارة الألمانية أنجيلا ميركل بفيروس، وخضوعها لحجر منزلي رغم أن نتائج الفحوص التي أجرتها كانت سلبية، ومع ذلك قررت الخضوع للحجر الصحي والبقاء في المنزل.
في دوائر السلطة وقصور الحكم وخلال الأسابيع الماضية ومع اتساع دائرة العدوى وعدد الإصابات بفيروس كورونا في عدة دول ومناطق عبر العالم، بدأ الفيروس تدريجيا يقترب من مراكز اتخاذ القرار، وربما يعرض حياة قادة ومسؤولين كبار للخطر.
ولاحقا، أعلنت السلطات البرازيلية أن الوزير الذي التقى ترامب مصاب بالفيروس، كما وضع الرئيس جايير بولسونارو تحت المراقبة الصحية للتأكد من عدم إصابته.
'''

rouge = Rouge()
scores = rouge.get_scores(generated_summary, summary_evaluation)
print(scores)
```


