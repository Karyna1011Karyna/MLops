#Demo helper
from common import db

NEW_EXAMPLES = [
    ("Чи варто мені почати навчання новій професії цього року?", "kariera"),
    ("Що заважає мені відпустити старі образи?", "dukhovnist"),
    ("Чи покращиться моє здоров'я після відпустки?", "zdorovia"),
    ("Чи варто позичати гроші другу?", "finansy"),
    ("Чи щирі почуття цієї людини до мене?", "kohannia"),
    ("Чи вдасться мені завершити цей проєкт вчасно?", "zagalne"),
    ("Чи варто мені шукати нового ментора в кар'єрі?", "kariera"),
    ("Що допоможе мені відновити довіру у стосунках?", "kohannia"),
    ("Чи варто змінити банк, у якому я тримаю заощадження?", "finansy"),
    ("Що я маю зробити, щоб знайти внутрішній спокій?", "dukhovnist"),
]


def main():
    ids = [db.insert_training_example(q, label, source="simulated_new_data") for q, label in NEW_EXAMPLES]
    print(f"Inserted {len(ids)} new labeled examples into training_examples: {ids}")


if __name__ == "__main__":
    main()
