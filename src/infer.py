from simpletransformers.question_answering import QuestionAnsweringModel


def main():

    model = QuestionAnsweringModel(
        'distilbert', 'distilbert-base-uncased-distilled-squad')

    print("Ready to roll")
    while True:
        result = model.predict([
            {
                'context':
                """Why A Higher Unemployment Rate Is Actually Good News This Time
                Obamacare Is More Unpopular Than Ever, Poll Shows
                U.S. Creates 209,000 Jobs In July, Unemployment Rate Rises To 6.2%
                It's Not About Leading; It's About Leading Well
                When experience and skills intersect with consistent values-based leadership style -- that is when the magic happens and greatness emerges.
                The Sharing Economy at a Crossroads
                """,
                'qas': [
                    {'id': '0', 'question': input()}
                ]
            }
        ], n_best_size=10)

        answers = result[0]
        probabilities = result[1]

        for i in range(len(answers)):
            pair = answers[i]
            for j in range(len(pair['answer'])):
                print(pair['id'] + ": " + pair['answer']
                      [j] + " - " + str(probabilities[i]['probability'][j]))
            print('\n')


if __name__ == '__main__':
    main()
