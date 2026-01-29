// ============================================================================
// MongoDB Data Cleaning Queries for ai_test_db
// Run these queries directly in MongoDB Compass, Studio 3T, or mongo shell
// ============================================================================

// Switch to your database
use("ai_test_db");

print("🚀 Starting MongoDB Data Cleaning Queries");
print("==========================================");

// ============================================================================
// 1. EMAIL CLEANING - Fix uppercase and trailing spaces
// ============================================================================

print("\n📧 1. CLEANING CUSTOMER EMAILS");
print("------------------------------");

// First, let's see the problematic emails
print("🔍 Finding problematic emails...");
var problemEmails = db.customers.find(
    { "email": { $regex: /[A-Z]|\s$/ } },
    { "email": 1, "_id": 0 }
).limit(5).toArray();

problemEmails.forEach(function(doc) {
    print("   • '" + doc.email + "'");
});

// Count problematic emails
var invalidEmailCount = db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } });
print("📊 Invalid emails found: " + invalidEmailCount);

// Clean emails - convert to lowercase and trim spaces
print("🧹 Cleaning emails...");
var emailUpdateResult = db.customers.updateMany(
    { "email": { $regex: /[A-Z]|\s/ } },
    [
        {
            $set: {
                "email": {
                    $trim: {
                        input: { $toLower: "$email" }
                    }
                }
            }
        }
    ]
);

print("   • Emails updated: " + emailUpdateResult.modifiedCount);

// Verify cleaning
var remainingInvalid = db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } });
print("✅ Invalid emails remaining: " + remainingInvalid);

// ============================================================================
// 2. DUPLICATE PRODUCT REMOVAL
// ============================================================================

print("\n📦 2. REMOVING DUPLICATE PRODUCTS");
print("----------------------------------");

// Find duplicate product names
print("🔍 Finding duplicate products...");
var duplicates = db.products.aggregate([
    {
        $group: {
            _id: "$name",
            ids: { $push: "$_id" },
            count: { $sum: 1 }
        }
    },
    {
        $match: { count: { $gt: 1 } }
    }
]).toArray();

print("📊 Duplicate product groups found: " + duplicates.length);

// Show examples
print("🔍 Examples of duplicates:");
duplicates.slice(0, 3).forEach(function(dup) {
    print("   • '" + dup._id + "' appears " + dup.count + " times");
});

// Process each duplicate group
var totalRemoved = 0;
var totalOrdersUpdated = 0;

duplicates.forEach(function(duplicate) {
    var productName = duplicate._id;
    var productIds = duplicate.ids;
    
    // Keep first product, remove others
    var keepId = productIds[0];
    var removeIds = productIds.slice(1);
    
    print("\n🔄 Processing '" + productName + "':");
    print("   • Keeping: " + keepId);
    print("   • Removing: " + removeIds.length + " duplicates");
    
    // Update orders to reference the kept product
    var updateResult = db.orders.updateMany(
        { "productId": { $in: removeIds } },
        { $set: { "productId": keepId } }
    );
    
    // Remove duplicate products
    var deleteResult = db.products.deleteMany(
        { "_id": { $in: removeIds } }
    );
    
    totalRemoved += deleteResult.deletedCount;
    totalOrdersUpdated += updateResult.modifiedCount;
    
    print("   • Orders updated: " + updateResult.modifiedCount);
    print("   • Products removed: " + deleteResult.deletedCount);
});

print("\n📈 DEDUPLICATION RESULTS:");
print("   • Total products removed: " + totalRemoved);
print("   • Total order references updated: " + totalOrdersUpdated);

// Verify no duplicates remain
var remainingDuplicates = db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]).toArray().length;

print("   • Duplicate names remaining: " + remainingDuplicates);

// ============================================================================
// 3. PHONE NUMBER STANDARDIZATION
// ============================================================================

print("\n📞 3. STANDARDIZING PHONE NUMBERS");
print("----------------------------------");

// Show current phone formats
print("🔍 Current phone number samples:");
var phonesamples = db.customers.find(
    { "phone": { $exists: true } },
    { "phone": 1, "_id": 0 }
).limit(5).toArray();

phonesamples.forEach(function(doc) {
    print("   • '" + doc.phone + "'");
});

// Standardize phone numbers - Simple approach
print("🧹 Standardizing phone numbers...");

// Get all customers with phone numbers
var customers = db.customers.find({ "phone": { $exists: true } }).toArray();
var phoneUpdated = 0;

customers.forEach(function(customer) {
    var originalPhone = customer.phone.toString();
    var digitsOnly = originalPhone.replace(/\D/g, ''); // Remove non-digits
    var formattedPhone = originalPhone; // Default to original
    
    // Format based on digit count
    if (digitsOnly.length === 10) {
        formattedPhone = "+91-" + digitsOnly.substring(0, 5) + "-" + digitsOnly.substring(5);
    } else if (digitsOnly.length === 12 && digitsOnly.startsWith('91')) {
        formattedPhone = "+" + digitsOnly.substring(0, 2) + "-" + digitsOnly.substring(2, 7) + "-" + digitsOnly.substring(7);
    }
    
    // Update if changed
    if (originalPhone !== formattedPhone) {
        db.customers.updateOne(
            { "_id": customer._id },
            { $set: { "phone": formattedPhone } }
        );
        phoneUpdated++;
    }
});

print("📈 Phone standardization results:");
print("   • Phone numbers processed: " + phoneUpdated);

// Show standardized phone samples
print("🔍 Standardized phone number samples:");
var newPhonesamples = db.customers.find(
    { "phone": { $exists: true } },
    { "phone": 1, "_id": 0 }
).limit(5).toArray();

newPhonesamples.forEach(function(doc) {
    print("   • '" + doc.phone + "'");
});

// ============================================================================
// 4. DATA QUALITY VERIFICATION
// ============================================================================

print("\n📊 4. FINAL DATA QUALITY VERIFICATION");
print("--------------------------------------");

// Count total records
var totalCustomers = db.customers.countDocuments({});
var totalProducts = db.products.countDocuments({});
var totalOrders = db.orders.countDocuments({});

print("📈 Database Statistics:");
print("   • Total customers: " + totalCustomers);
print("   • Total products: " + totalProducts);
print("   • Total orders: " + totalOrders);

// Check email quality
var invalidEmails = db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } });
print("   • Invalid emails: " + invalidEmails);

// Check product duplicates
var productDuplicates = db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]).toArray().length;
print("   • Duplicate product names: " + productDuplicates);

// Check orphaned orders (orders with non-existent products)
var orphanedOrders = db.orders.aggregate([
    {
        $lookup: {
            from: "products",
            localField: "productId",
            foreignField: "_id",
            as: "product"
        }
    },
    {
        $match: { "product": { $size: 0 } }
    },
    {
        $count: "orphanedCount"
    }
]).toArray();

var orphanCount = orphanedOrders.length > 0 ? orphanedOrders[0].orphanedCount : 0;
print("   • Orphaned orders: " + orphanCount);

print("\n✅ DATA CLEANING COMPLETED!");
print("============================");

if (invalidEmails === 0 && productDuplicates === 0 && orphanCount === 0) {
    print("🎉 Perfect! Your database is now clean and consistent.");
} else {
    print("⚠️  Some issues may remain. Review the counts above.");
}

print("\n💡 BACKUP RECOMMENDATION:");
print("Consider creating a backup of your cleaned database:");
print("mongodump --db ai_test_db --out ./backup_cleaned_" + new Date().toISOString().slice(0,10));