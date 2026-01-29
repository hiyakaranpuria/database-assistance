// Simple MongoDB Cleaning Queries - Copy and paste these one by one
// ================================================================

// 1. CLEAN CUSTOMER EMAILS
// ========================

// Check invalid emails before cleaning
print("Invalid emails before cleaning:");
db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } });

// Show examples
db.customers.find({ "email": { $regex: /[A-Z]|\s$/ } }, { "email": 1 }).limit(3);

// Clean emails (convert to lowercase and trim)
db.customers.updateMany(
    { "email": { $regex: /[A-Z]|\s/ } },
    [{ $set: { "email": { $trim: { input: { $toLower: "$email" } } } } }]
);

// Check invalid emails after cleaning (should be 0)
print("Invalid emails after cleaning:");
db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } });

// 2. REMOVE DUPLICATE PRODUCTS
// ============================

// Check duplicates before cleaning
print("Duplicate products before cleaning:");
db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]);

// Get total products before
print("Total products before:");
db.products.countDocuments({});

// Remove duplicates (keep first, remove others)
var duplicates = db.products.aggregate([
    { $group: { _id: "$name", ids: { $push: "$_id" }, count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]).toArray();

duplicates.forEach(function (dup) {
    var keepId = dup.ids[0];
    var removeIds = dup.ids.slice(1);

    // Update orders to reference kept product
    db.orders.updateMany(
        { "productId": { $in: removeIds } },
        { $set: { "productId": keepId } }
    );

    // Remove duplicate products
    db.products.deleteMany({ "_id": { $in: removeIds } });

    print("Processed: " + dup._id);
});

// Check duplicates after cleaning (should be empty)
print("Duplicate products after cleaning:");
db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]);

// Get total products after
print("Total products after:");
db.products.countDocuments({});

// 3. STANDARDIZE PHONE NUMBERS
// ============================

// Show phone samples before
print("Phone samples before:");
db.customers.find({ "phone": { $exists: true } }, { "phone": 1 }).limit(3);

// Standardize phones (simple JavaScript approach)
db.customers.find({ "phone": { $exists: true } }).forEach(function (customer) {
    var phone = customer.phone.toString();
    var digits = phone.replace(/\D/g, '');
    var formatted = phone;

    if (digits.length === 10) {
        formatted = "+91-" + digits.substring(0, 5) + "-" + digits.substring(5);
    } else if (digits.length === 12 && digits.startsWith('91')) {
        formatted = "+" + digits.substring(0, 2) + "-" + digits.substring(2, 7) + "-" + digits.substring(7);
    }

    if (phone !== formatted) {
        db.customers.updateOne(
            { "_id": customer._id },
            { $set: { "phone": formatted } }
        );
    }
});

// Show phone samples after
print("Phone samples after:");
db.customers.find({ "phone": { $exists: true } }, { "phone": 1 }).limit(3);

// 4. FINAL VERIFICATION
// ====================

print("=== FINAL RESULTS ===");
print("Invalid emails: " + db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } }));
print("Duplicate products: " + db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]).toArray().length);
print("Total customers: " + db.customers.countDocuments({}));
print("Total products: " + db.products.countDocuments({}));
print("Total orders: " + db.orders.countDocuments({}));

print("✅ Cleaning completed!");