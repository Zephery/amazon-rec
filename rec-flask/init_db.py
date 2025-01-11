from db.amazon_categories import init_categories
from db.amazon_products import init_products
from db.amazon_reviews import init_review

if __name__ == '__main__':
    init_categories()
    init_products()
    init_review()
