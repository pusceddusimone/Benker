import os

from matplotlib import pyplot as plt

import classifier
import numpy as np
import telebot
from telebot import types
import preprocessing as pp
import random
import pandas as pd

BOT_TOKEN = '6178214879:AAGYzkr93SJzPkf3YGgLPt6pnqanQCj1kq4'

bot = telebot.TeleBot(BOT_TOKEN)

array_last_predictions = []

model = vectorizer = None

button_yes = types.InlineKeyboardButton('Yes ✅', callback_data='button_yes')
button_no = types.InlineKeyboardButton('No ❌', callback_data='button_no')

keyboard = types.InlineKeyboardMarkup()
keyboard.add(button_yes)
keyboard.add(button_no)

last_bot_message = None
last_user_message = None

label_intent_mapping = {
    0: "Activate My Card",
    1: "Age Limit",
    2: "Apple Pay or Google Pay",
    3: "ATM Support",
    4: "Automatic Top Up",
    5: "Balance Not Updated After Bank Transfer",
    6: "Balance Not Updated After Cheque or Cash Deposit",
    7: "Beneficiary Not Allowed",
    8: "Cancel Transfer",
    9: "Card About to Expire",
    10: "Card Acceptance",
    11: "Card Arrival",
    12: "Card Delivery Estimate",
    13: "Card Linking",
    14: "Card Not Working",
    15: "Card Payment Fee Charged",
    16: "Card Payment Not Recognised",
    17: "Wrong Exchange Rate In Card Payment",
    18: "Card Swallowed",
    19: "Cash Withdrawal Charge",
    20: "Cash Withdrawal Not Recognised",
    21: "Change PIN",
    22: "Compromised Card",
    23: "Contactless Not Working",
    24: "Country Support",
    25: "Declined Card Payment",
    26: "Declined Cash Withdrawal",
    27: "Declined Transfer",
    28: "Direct Debit Payment Not Recognised",
    29: "Disposable Card Limits",
    30: "Edit Personal Details",
    31: "Exchange Charge",
    32: "Exchange Rate",
    33: "Exchange via App",
    34: "Extra Charge on Statement",
    35: "Failed Transfer",
    36: "Fiat Currency Support",
    37: "Get Disposable Virtual Card",
    38: "Get Physical Card",
    39: "Getting Spare Card",
    40: "Getting Virtual Card",
    41: "Lost or Stolen Card",
    42: "Lost or Stolen Phone",
    43: "Order Of Physical Card",
    44: "Passcode Forgotten",
    45: "Pending Card Payment",
    46: "Pending Cash Withdrawal",
    47: "Pending Top Up",
    48: "Pending Transfer",
    49: "PIN Blocked",
    50: "Receiving Money",
    51: "Refund Not Showing Up",
    52: "Request Refund",
    53: "Reverted Card Payment?",
    54: "Supported Cards and Currencies",
    55: "Terminate Account",
    56: "Top Up by Bank Transfer Charge",
    57: "Top Up by Card Charge",
    58: "Top Up by Cash or Cheque",
    59: "Top Up Failed",
    60: "Top Up Limits",
    61: "Top Up Reverted",
    62: "Topping Up by Card",
    63: "Transaction Charged Twice",
    64: "Transfer Fee Charged",
    65: "Transfer into Account",
    66: "Transfer Not Received by Recipient",
    67: "Transfer Timing",
    68: "Unable to Verify Identity",
    69: "Verify My Identity",
    70: "Verify Source of Funds",
    71: "Verify Top Up",
    72: "Virtual Card Not Working",
    73: "Visa or Mastercard",
    74: "Why Verify Identity",
    75: "Wrong amount of cash received",
    76: "Wrong Exchange Rate For Cash Withdrawal"
}


def check_if_model_trained():
    return model is None or vectorizer is None


def from_number_to_class(num):
    if num in label_intent_mapping:
        return label_intent_mapping[num]
    else:
        "Don't know"


def train_model(chat_id, chosen_model):
    unified_message_sender(chat_id, "Superb choice, hold on a second while i train the model Monsieur")
    global model, vectorizer
    if chosen_model == 'SVM':
        model, vectorizer = classifier.train_model_svm()
    elif chosen_model == 'LR':
        model, vectorizer = classifier.train_model_logistic_regression()
    elif chosen_model == 'NB':
        model, vectorizer = classifier.train_model_naive_bayes()
    unified_message_sender(chat_id,
                           "And... done!\nWrite any bank related request and I'll try to answer as best as I can")


def unified_message_sender(chat_id, message, keyboard_markup=None, parse_mode=None):
    global last_bot_message
    sent_message = bot.send_message(chat_id, message, reply_markup=keyboard_markup, parse_mode=parse_mode)
    last_bot_message = sent_message


def remove_buttons(message):
    if message is not None and message.reply_markup is not None:
        global last_bot_message
        last_bot_message = bot.edit_message_reply_markup(message.chat.id, message.message_id, reply_markup=None)


def get_random_response(prediction):
    responses = [
        "Mmmh let's see now...\nThe crystal ball say's it's <u><b>{}</b></u>!🔮\nIs it correct?",
        "Ah, this one is too easy, it's <u><b>{}</b></u>, right?",
        "Pretty sure it's <u><b>{}</b></u>, yes?",
        "It must be <u><b>{}</b></u>, am I right?",
    ]
    response_template = random.choice(responses)
    return response_template.format(prediction)


def register_user_feedback(text, label):
    try:
        new_element = [text, label]
        file_path = os.path.join(os.path.dirname(__file__), 'dataset', 'banking-training-user.csv')
        new_dataframe = pd.DataFrame([new_element])
        new_dataframe.to_csv(file_path, mode='a', header=False, index=False)

    except FileNotFoundError:
        print("Parquet file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    keyboard_greet = types.InlineKeyboardMarkup()
    button_svm = types.InlineKeyboardButton('Support Vector Machine', callback_data='SVM')
    button_log_regr = types.InlineKeyboardButton('Logistic Regression', callback_data='LR')
    button_naive_bayes = types.InlineKeyboardButton('Naive Bayes', callback_data='NB')
    keyboard_greet.add(button_svm)
    keyboard_greet.add(button_log_regr)
    keyboard_greet.add(button_naive_bayes)
    unified_message_sender(message.chat.id, "Howdy, choose the classifier that you'd like to use",
                           keyboard_markup=keyboard_greet)


@bot.message_handler(commands=['stop'])
def send_goodbye(message):
    unified_message_sender(message.chat.id, "Farewell")
    bot.stop_polling()


@bot.message_handler(commands=['report'])
def send_report(message):
    if check_if_model_trained():
        unified_message_sender(message.chat.id, "Looks like you haven't trained a model yet\nType /start to train one")
        return
    unified_message_sender(message.chat.id, "Report? Very well, just one second while I generate it")
    data = classifier.generate_report(model, vectorizer)
    df = pd.DataFrame(data).T
    excluded_indices = list(df.index)[-3:]
    df = df.drop(excluded_indices)
    df.index = df.index.astype(int)
    df.index = df.index.map(label_intent_mapping)
    excluded_df = pd.DataFrame(data).T.loc[excluded_indices]
    df = pd.concat([df, excluded_df])
    df = df.round(decimals=2)
    table_plot = plt.table(cellText=df.values,
                           colLabels=df.columns,
                           rowLabels=df.index,
                           cellLoc='center',
                           loc='center')

    table_plot.auto_set_font_size(False)
    table_plot.set_fontsize(12)
    table_plot.scale(1.2, 1.2)

    plt.axis('off')

    table_path = os.path.join(os.path.dirname(__file__), 'images', 'table_image.png')
    plt.savefig(table_path, bbox_inches='tight')
    with open(table_path, 'rb') as image_file:
        bot.send_photo(message.chat.id, image_file)


@bot.message_handler(commands=['accuracy'])
def send_accuracy(message):
    if check_if_model_trained():
        unified_message_sender(message.chat.id, "Looks like you haven't trained a model yet\nType /start to train one")
        return
    unified_message_sender(message.chat.id, "Hold on a second, let me search where I left it...")
    unified_message_sender(message.chat.id,
                           "There it is! \nMy accuracy is: {}".format(classifier.calculate_accuracy(model, vectorizer)))


@bot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    global last_bot_message
    remove_buttons(last_bot_message)
    if call.data in ['SVM', 'LR', 'NB']:
        train_model(call.message.chat.id, call.data)
    elif call.data == 'button_yes':
        unified_message_sender(call.message.chat.id, "Oh wow, I mean obviously I'm right\n"
                                                     "Do you have more requests?")
    elif call.data == 'button_no':
        global array_last_predictions
        sorted_indices = np.argsort(array_last_predictions[0])[::-1]
        keyboard_query = types.InlineKeyboardMarkup()
        max_options_number = 5
        for index in range(1, max_options_number):
            button = types.InlineKeyboardButton(from_number_to_class(sorted_indices[index]),
                                                callback_data=str(sorted_indices[index]))
            keyboard_query.add(button)
        button = types.InlineKeyboardButton("Not one of these ❌",
                                            callback_data="wrong_answer")
        keyboard_query.add(button)
        unified_message_sender(call.message.chat.id,
                               "Ouch, maybe your request is one of these?",
                               keyboard_query)
    else:
        global last_user_message
        if last_user_message is not None and call.data != 'wrong_answer':
            register_user_feedback(call.data, last_user_message.text)
            unified_message_sender(last_user_message.chat.id, "Not all heroes wear capes, but I have one that fits "
                                                              "you well\n"
                                                              "The response has been recorded and it will take effect "
                                                              "when you will execute /start again. \n"
                                                              "Thanks for the feedback💅")
        else:
            unified_message_sender(last_user_message.chat.id, "I give up 🏳.️\nCan I try with "
                                                              "another one?")


@bot.message_handler(func=lambda msg: True)
def answer_request(message):
    if check_if_model_trained():
        unified_message_sender(message.chat.id, "Looks like you haven't trained a model yet\nType /start to train one")
        return
    global last_user_message
    last_user_message = message
    text_preprocessed = pp.preprocess(message.text)
    text_preprocessed = text_preprocessed["message"][0]
    text_preprocessed = [' '.join(text_preprocessed)]
    text_to_predict_transformed = vectorizer.transform(text_preprocessed).todense()
    text_to_predict_transformed = np.array(text_to_predict_transformed)

    y_pred = model.predict(text_to_predict_transformed)
    y_probabilities = model.predict_proba(text_to_predict_transformed)
    global array_last_predictions
    global last_bot_message
    remove_buttons(last_bot_message)
    array_last_predictions = y_probabilities
    sorted_array = np.sort(array_last_predictions)[::-1]
    text_prediction = from_number_to_class(y_pred[0])
    text_response = get_random_response(text_prediction)
    unified_message_sender(message.chat.id,
                           "{}\nProbability of {}%".format(text_response, round(sorted_array[0][-1] * 100, 2)),
                           parse_mode='HTML', keyboard_markup=keyboard)


bot.infinity_polling()
