# Individual MongoDB Cleaning Queries

You can run these queries one by one in MongoDB Compass, Studio 3T, or mongo shell.

## 🔍 1. ANALYZE DATA QUALITY ISSUES

### Find Invalid Emails
```javascript
// Count invalid emails (uppercase or trailing spaces)
db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } })

// Show examples of invalid emails
db.customers.find(
    { "email": { $regex: /[A-Z]|\s$/ } },
    { "email": 1, "name": 1 }
).limit(10)
```

### Find Duplicate Products
```javascript
// Find duplicate product names
db.products.aggregate([
    {
        $group: {
            _id: "$name",
            count: { $sum: 1 },
            ids: { $push: "$_id" }
        }
    },
    {
        $match: { count: { $gt: 1 } }
    },
    {
        $sort: { count: -1 }
    }
])
```

### Analyze Phone Number Formats
```javascript
// Show different phone number formats
db.customers.aggregate([
    {
        $group: {
            _id: { $strLenCP: { $toString: "$phone" } },
            count: { $sum: 1 },
            examples: { $push: "$phone" }
        }
    },
    {
        $project: {
            length: "$_id",
            count: 1,
            examples: { $slice: ["$examples", 3] }
        }
    }
])
```

## 🧹 2. EXECUTE CLEANING OPERATIONS

### Clean Customer Emails
```javascript
// Fix uppercase and trailing spaces in emails
db.customers.updateMany(
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
)
```

### Remove Duplicate Products (Step by Step)

#### Step 1: Find one specific duplicate to test
```javascript
// Find a specific duplicate product
var testDuplicate = db.products.aggregate([
    { $group: { _id: "$name", ids: { $push: "$_id" }, count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } },
    { $limit: 1 }
]).toArray()[0];

print("Testing with product: " + testDuplicate._id);
print("IDs to process: " + testDuplicate.ids);
```

#### Step 2: Update orders for this duplicate
```javascript
// Keep first ID, update orders pointing to others
var keepId = testDuplicate.ids[0];
var removeIds = testDuplicate.ids.slice(1);

// Update orders to point to the kept product
db.orders.updateMany(
    { "productId": { $in: removeIds } },
    { $set: { "productId": keepId } }
)
```

#### Step 3: Remove duplicate products
```javascript
// Remove the duplicate products
db.products.deleteMany({ "_id": { $in: removeIds } })
```

#### Step 4: Process ALL duplicates (run after testing)
```javascript
// Process all duplicates at once
var duplicates = db.products.aggregate([
    { $group: { _id: "$name", ids: { $push: "$_id" }, count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
]).toArray();

duplicates.forEach(function(dup) {
    var keepId = dup.ids[0];
    var removeIds = dup.ids.slice(1);
    
    // Update orders
    db.orders.updateMany(
        { "productId": { $in: removeIds } },
        { $set: { "productId": keepId } }
    );
    
    // Remove duplicates
    db.products.deleteMany({ "_id": { $in: removeIds } });
    
    print("Processed: " + dup._id + " (removed " + removeIds.length + " duplicates)");
});
```

### Standardize Phone Numbers
```javascript
// Standardize phone number formats
db.customers.updateMany(
    { "phone": { $exists: true } },
    [
        {
            $set: {
                "phone": {
                    $let: {
                        vars: {
                            digitsOnly: {
                                $replaceAll: {
                                    input: { $toString: "$phone" },
                                    find: { $regex: /\D/ },
                                    replacement: ""
                                }
                            }
                        },
                        in: {
                            $switch: {
                                branches: [
                                    {
                                        case: { $eq: [{ $strLenCP: "$$digitsOnly" }, 10] },
                                        then: {
                                            $concat: [
                                                "+91-",
                                                { $substr: ["$$digitsOnly", 0, 5] },
                                                "-",
                                                { $substr: ["$$digitsOnly", 5, 5] }
                                            ]
                                        }
                                    },
                                    {
                                        case: {
                                            $and: [
                                                { $eq: [{ $strLenCP: "$$digitsOnly" }, 12] },
                                                { $eq: [{ $substr: ["$$digitsOnly", 0, 2] }, "91"] }
                                            ]
                                        },
                                        then: {
                                            $concat: [
                                                "+",
                                                { $substr: ["$$digitsOnly", 0, 2] },
                                                "-",
                                                { $substr: ["$$digitsOnly", 2, 5] },
                                                "-",
                                                { $substr: ["$$digitsOnly", 7, 5] }
                                            ]
                                        }
                                    }
                                ],
                                default: "$phone"
                            }
                        }
                    }
                }
            }
        }
    ]
)
```

## 📊 3. VERIFY RESULTS

### Check Email Cleaning Results
```javascript
// Should return 0 if cleaning worked
db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } })

// Show cleaned email samples
db.customers.find({}, { "email": 1, "_id": 0 }).limit(5)
```

### Check Product Deduplication Results
```javascript
// Should return empty array if no duplicates remain
db.products.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $match: { count: { $gt: 1 } } }
])

// Check total product count
db.products.countDocuments({})
```

### Check Phone Standardization Results
```javascript
// Show standardized phone samples
db.customers.find(
    { "phone": { $exists: true } },
    { "phone": 1, "_id": 0 }
).limit(10)
```

### Check for Orphaned Orders
```javascript
// Find orders pointing to non-existent products
db.orders.aggregate([
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
        $count: "orphanedOrders"
    }
])
```

## 🛡️ 4. SAFETY COMMANDS

### Create Backup Before Cleaning
```javascript
// Create backup collections
db.customers.aggregate([{ $out: "customers_backup_" + new Date().toISOString().slice(0,10) }])
db.products.aggregate([{ $out: "products_backup_" + new Date().toISOString().slice(0,10) }])
db.orders.aggregate([{ $out: "orders_backup_" + new Date().toISOString().slice(0,10) }])
```

### Restore from Backup (if needed)
```javascript
// Restore customers (replace YYYY-MM-DD with your backup date)
db.customers.drop()
db.customers_backup_YYYY-MM-DD.aggregate([{ $out: "customers" }])

// Restore products
db.products.drop()
db.products_backup_YYYY-MM-DD.aggregate([{ $out: "products" }])

// Restore orders
db.orders.drop()
db.orders_backup_YYYY-MM-DD.aggregate([{ $out: "orders" }])
```

## 🚀 Quick Start Commands

Run these in order for complete cleaning:

```javascript
// 1. Create backups
db.customers.aggregate([{ $out: "customers_backup" }])
db.products.aggregate([{ $out: "products_backup" }])
db.orders.aggregate([{ $out: "orders_backup" }])

// 2. Clean emails
db.customers.updateMany({ "email": { $regex: /[A-Z]|\s/ } }, [{ $set: { "email": { $trim: { input: { $toLower: "$email" } } } } }])

// 3. Remove duplicates (run the full duplicate removal script above)

// 4. Standardize phones (run the phone standardization script above)

// 5. Verify results
db.customers.countDocuments({ "email": { $regex: /[A-Z]|\s$/ } })
db.products.aggregate([{ $group: { _id: "$name", count: { $sum: 1 } } }, { $match: { count: { $gt: 1 } } }])
```