import mysql.connector

mydb = mysql.connector.connect(
	host= "localhost",
	user="username",
	password="password"
#	database ="defi"
	)
cursor= mycursor()
mycursor.execute("CREATE TABLE Users (user_id number(16) primary key, first_name varchar (20), last_name varchar (20), id_no number(10), email varchar(50), KRA_pin varchar (10), password varchar(20), pin number(4)) ")

cur.execute('CREATE TABLE Account(acc_no number(16) primary key,user_id number(16),password varchar2(50),type varchar2(1),balance number(10),withdrawl_count number(2),last_date date,closure_date date)')
print("Account Table Created Successfully...\n")

cur.execute('CREATE TABLE Transactions(transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,account_no number(16),type varchar2(1),transaction_time date,balance number(10),amount number(10))')
print("Transaction Table Created Successfully...\n")

db.close()

