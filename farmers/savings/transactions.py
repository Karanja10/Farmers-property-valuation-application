from users import User
from transfer import Shares
from accounts import Account


count = 0
error_type = None


user=User()

print("Welcome to transaction Audits")
print("Please provide the file name")

file= str(input("File name:"))


count=user.processTransactionLog(file, error_type)


if file=="F":
	print("ERROR: File not found")
elif file== "C":
	print("ERROR: Corruption detected in transaction log transaction aborted")

	print("SUCCESS")

print(f"First Name {user.getFirst_name()}")
print(f"Account No {user.acc_num()}")
print(f"Transaction count {count()}")
print(f"Checking balance {user.getCheckingBalance()}")
print(f"Savings balance {user.getSavingsingBalance()}")
print(f"Shares balance{user.getSharesBalance()}")

