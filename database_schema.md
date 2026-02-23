Collection 'products': {name(string), price(number), categoryId(unknown), stock(number), createdAt(date)}
Collection 'reviews': {productId(unknown), rating(number), comment(string), createdAt(date)}
Collection 'payments': {orderId(unknown), amount(number), method(string), status(string), paymentDate(date)}
Collection 'users': {name(string), email(string), role(string), createdAt(date), categoryId(unknown)}
Collection 'orders': {customerId(unknown), productId(unknown), quantity(number), amount(number), status(string), orderDate(date)}
Collection 'categories': {name(string), createdAt(date)}
Collection 'customers': {name(string), email(string), phone(string), city(string), createdAt(date)}