- 👋 Hi, I’m @rohanmori2004gmail-com
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
rohanmori2004gmail-com/rohanmori2004gmail-com is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
INITIAL SETUP

Step 1:- create virtual environment
python -m venv myenv

use myenv or active myenv
Set-ExecutionPolicy RemoteSigned -Scope Process myenv/Scripts/activate

Step:2 install all libraries/dependencies
pip install torch pip install nltk pip install flask pip install json pip install torchvision or Pip install -r requirements.txt

Install nltk package
(myenv) python

 import nltk nltk.download('punkt')

            or  
#first time nltk_util.py file run remove "nltk.download('punkt')" from comment; after successfully execute make it comment

Step:3 Modify intents.json
Modify intents.json with different intents and responses for your Chatbot

Step 4 :- RUN
first train your chatbot
(myenv) python train.py

after run chat.py file (this file run o terminal you can chat here)
(myenv) python chat.py

for flask web application /UI
(myenv) python app.py

(don't close app.py file) Go live with base.html file
