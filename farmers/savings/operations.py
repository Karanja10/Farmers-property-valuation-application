import numpy as np 
import pandas as pd 
import datetime
from savings.accounts import Account
from savings.users import User
from savings.transfer import Shares
import mysql.connector
import database

def defi_account(acc_no, acc_balance, acc_type):
	return Account(acc_no, acc_balance, acc_type, amount)
def defi_user(first_name, last_name, email, pin, password):
	return User(first_name, last_name,email,pin,password)


def log():
	pass


def logging_in():
	for i in range(0,3):
		try:
			global acc
			account= int(input("acc_no:"))
			password=input("password:")
		except:
			continue

		try:
			mydb=cursor.execute("SELECT password FROM accounts WHERE acc_no = 1: and closure date = NULL",(acc,))
		except: 
    		print("Database error")
		continue

			mydb=cursor.fetchall()
			if (mydb):
				mydb=mydb[0][0]
				if (mydb==password)
				loginpage()
				break
				else:
					print("invalid password")

				else:
					print("Invalid account number")
def signUp():
	for i in range(0,11):
		acc_type=int(input("Select type of account S for savings and H for shares :" ))
		amount=int(input("Initial amount"))
		if (acc is H and amount<5000) or (amount<0):
			print("Invalid account try savings")

			continue

			print("Enter your detals here")

			first_name=input("First Name :")
			last_name=input("Last Name:")
			id_no=input("ID No:")
			KRA_pin=input("KRA:")
			email=input("Email:")
			password=input("password:")

			try:
				pincode=input("pincode\n")
			except:
				print("invalid pincode")

			continue

			from random import randint 
			rand= randint(1000,5000)

			cursor.execute("INSERT INTO users VALUES (:0,:1,:2,:3,:4,:5,:6)",first_name,last_name,id_no,KRA_pin,email,password,pincode)

			db.commit()

			ano =cursor.execute('SELECT user_id FROM users WHERE first_name =:1, and address =:2' (first_name, address))
			acc =cursor.fetchall()
			acc=acc[0][0]


			password=input("Enter password :" )

			cur.execute('INSERT INTO Account VALUES(:0,:1,:2,:3,:4,0,:5,NULL)',(rand,acc,password,acc_type,amount,datetime.datetime.now()))

            db.commit()

            acc_no = cur.execute('SELECT account_no FROM Account where customer_id = :1 and password = :2',(acc_no, password)).fetchall()
            acn = acc_no[0][0]

            print ("\nYour Account number : ",acc_no)

            break


    def loginpage():
    user_id = cur.execute('SELECT user_id FROM Account WHERE account_no = :1',(acc,)).fetchall()
    user_id = cus_id[0][0]
    for i in range(0,10):
        print("Choose Wisely\n\n1.Balance enquiry \n2. Address Change\n3. Money Deposit\n4. Money Withdrawal\n5. Transfer Money\n6. Account Closure\n7. View Profile\n0. Customer Logout")
        x = int(input("Enter your choice : "))
        if(x is 1):
            bal=cur.execute('SELECT balance from Account WHERE account_no=:1',(acc_no,)).fetchall()
            bal=bal[0][0]
            print("Your Current Balance is: ",bal)
        elif(x is 2):
            #Changing Address
            address = input("Enter address : ")
            city = input("City : ")
            state = input("county : ")
            pincode = int(input("Pincode : "))
        
            cur.execute('UPDATE Customer SET address = :1, city = :2, state = :3, pincode = :4 WHERE customer_id = :5',(address,city,county,pincode,cus_id))
            print("Address successfully changed\n")
            db.commit()

            








			
