class BankAccount(object):
 def __init__(self, amount = 0):
 self.amount = amount
 
 def is_negative_balance(self): 
 return self.amount < 0
 
# I just opened a new account with $100
my_account = BankAccount(100)
 
# Let's see if I'm in debt
if my_account.is_negative_balance:
 print "Oh no, I owe %s!" % my_account.amount
else:
 print "Good, my finances are in tact."
