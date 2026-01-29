from pymongo import MongoClient
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["ai_test_db"]

# Clear existing data
db.users.delete_many({})
db.customers.delete_many({})
db.products.delete_many({})
db.categories.delete_many({})
db.orders.delete_many({})
db.payments.delete_many({})
db.reviews.delete_many({})

# ---------------- USERS ----------------
users = []
for _ in range(200):
    users.append({
        "name": fake.name(),
        "email": fake.email().lower(),
        "role": random.choice(["admin", "staff", "manager"]),
        "createdAt": fake.date_time_between(start_date="-2y", end_date="now")
    })
db.users.insert_many(users)

# ---------------- CUSTOMERS ----------------
customers = []
for _ in range(1000):
    email = fake.email()
    # introduce duplicates & bad formatting
    if random.random() < 0.1:
        email = email.upper() + " "

    customers.append({
        "name": fake.name(),
        "email": email,
        "phone": random.choice([
            fake.phone_number(),
            fake.msisdn(),
            "+91-" + fake.msisdn()
        ]),
        "city": random.choice(["Udaipur", "Jaipur", "Delhi", "Mumbai"]),
        "createdAt": fake.date_time_between(start_date="-3y", end_date="now")
    })
db.customers.insert_many(customers)

customer_ids = list(db.customers.find({}, {"_id": 1}))

# ---------------- CATEGORIES ----------------
categories = []
for name in ["Electronics", "Clothing", "Furniture", "Books", "Groceries"]:
    categories.append({
        "name": name,
        "createdAt": datetime.now()
    })
db.categories.insert_many(categories)

category_ids = list(db.categories.find({}, {"_id": 1}))

# ---------------- PRODUCTS ----------------
products = []
for _ in range(300):
    products.append({
        "name": fake.word().capitalize(),
        "price": random.randint(100, 5000),
        "categoryId": random.choice(category_ids)["_id"],
        "stock": random.randint(0, 200),
        "createdAt": fake.date_time_between(start_date="-2y", end_date="now")
    })
db.products.insert_many(products)

product_ids = list(db.products.find({}, {"_id": 1, "price": 1}))

# ---------------- ORDERS ----------------
orders = []
for _ in range(5000):
    product = random.choice(product_ids)
    quantity = random.randint(1, 5)
    amount = product["price"] * quantity

    order_date = fake.date_time_between(start_date="-2y", end_date="now")

    orders.append({
        "customerId": random.choice(customer_ids)["_id"],
        "productId": product["_id"],
        "quantity": quantity,
        "amount": amount,
        "status": random.choice(["completed", "pending", "cancelled"]),
        "orderDate": order_date
    })
db.orders.insert_many(orders)

order_ids = list(db.orders.find({}, {"_id": 1, "amount": 1}))

# ---------------- PAYMENTS ----------------
payments = []
for order in random.sample(order_ids, 3500):
    payments.append({
        "orderId": order["_id"],
        "amount": order["amount"],
        "method": random.choice(["card", "upi", "netbanking", "cod"]),
        "status": random.choice(["success", "failed"]),
        "paymentDate": fake.date_time_between(start_date="-2y", end_date="now")
    })
db.payments.insert_many(payments)

# ---------------- REVIEWS ----------------
reviews = []
for _ in range(1200):
    reviews.append({
        "productId": random.choice(product_ids)["_id"],
        "rating": random.randint(1, 5),
        "comment": fake.sentence(),
        "createdAt": fake.date_time_between(start_date="-1y", end_date="now")
    })
db.reviews.insert_many(reviews)

print("✅ Sample MongoDB data generated successfully!")
