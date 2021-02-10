from savings.users import Users
from savings.accounts import Account


class Shares(Account):
		balance = 0
		interest= 0
    def __init__(self,b, i, value):
		self.balance=float(b)
		self.interest=float(i)
		super(). __init__(acc_no,acc_balance,acc_type)
		self.acc_no=acc_no
		self.acc_balance=acc_balance
		self.acc_type=acc_type

	def share_purchase():
		self.balance += num

	def withdraw_():
		if (num <= self.balance):
			return True
			else:
				self.balance -= 10
				return False
	def transfer():
		if (withdraw_(num)):
			share_purchase(num)
			return True
		else:
			self.balance -= 10
			return False
	def postInterest(interest, customer):
		balance += balance * (interest / 12)
	def setInterest():
		interest = interestIn / 100
	def getValue():
		return self.balance