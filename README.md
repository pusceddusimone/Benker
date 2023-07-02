# About Benker
My @ in telegram is [@Benker_bot](https://telegram.me/Benker_bot) !<br/>
Benker is born as a college project and it is a telegram bot which can classify textual requests of banking nature.<br/>
Benker provides 3 models of text classification, which are:
<ul>
  <li>Naive Bayes </li>
  <li>Logistic Regression</li>
  <li>Support Vector Machine</li>
</ul>
<h2>Scripts</h2>
Here's a brief description of all the scripts included in the project:
<ul>
  <li><ins><b>bot.py</b></ins>: used for all the bot's prompts and acts as an interface to the user via telegram</li>
  <li><ins><b>classifier.py</b></ins>: used to train the models and the vectorizers</li>
</ul>
<h2>Modules required</h2>
Before executing the program, you must install the following mandatory modules:
<ul>
  <li>pandas</li>
  <li>scikit-learn</li>
  <li>telebot</li>
  <li>matplotlib</li>
  <li>nltk</li>
  <li>pyarrow</li>
  <li>fastparquet</li>
</ul>
After you've set up all this modules, you just need to run the script bot.py and you're ready to go!

In order to run the bot you must first run the /start command where you will choose which of the 3 templates to use.
After choosing the model will be trained using the training set and after the training phase, you can start to make requests.

<h2>List of useful commands</h2>
<ul>
  <li><ins>/start</ins> starts the bot and lets you choose one of the previously mentioned models, you can always run /start to change the model that you're using</li>
  <li><ins>/accuracy</ins> sends the model's accuracy</li>
  <li><ins>/report</ins> generates a detailed report with the <b>precision, support, f1-score</b> and <b>recall</b> for each class and the average of these scores</li>
  <li><ins>/stop</ins> stops the bot</li>
</ul>
