from codebleu import calc_codebleu

prediction = """
import java.util.ArrayList;
import java.util.List;

// 1. Account class
class Account {
    private String accountNumber;
    private String owner;
    private double balance;

    public Account(String accountNumber, String owner, double balance) {
        this.accountNumber = accountNumber;
        this.owner = owner;
        this.balance = balance;
    }

    public String getAccountNumber() {
        return accountNumber;
    }

    public double getBalance() {
        return balance;
    }

    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            System.out.println(amount + " deposited to " + accountNumber);
        } else {
            System.out.println("Deposit amount must be positive.");
        }
    }

    public void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            System.out.println(amount + " withdrawn from " + accountNumber);
        } else if (amount > balance) {
            System.out.println("Insufficient balance for withdrawal.");
        } else {
            System.out.println("Withdrawal amount must be positive.");
        }
    }

    @Override
    public String toString() {
        return "Account{" +
                "accountNumber='" + accountNumber + '\'' +
                ", owner='" + owner + '\'' +
                ", balance=" + balance +
                '}';
    }
}

// 2. Bank class
class Bank {
    private List<Account> accounts;

    public Bank() {
        accounts = new ArrayList<>();
    }

    public void addAccount(Account account) {
        accounts.add(account);
        System.out.println("Account " + account.getAccountNumber() + " added.");
    }

    public Account findAccount(String accountNumber) {
        for (Account account : accounts) {
            if (account.getAccountNumber().equals(accountNumber)) {
                return account;
            }
        }
        System.out.println("Account " + accountNumber + " not found.");
        return null;
    }

    public void transfer(String fromAccountNumber, String toAccountNumber, double amount) {
        Account from = findAccount(fromAccountNumber);
        Account to = findAccount(toAccountNumber);

        if (from != null && to != null) {
            if (amount > 0 && from.getBalance() >= amount) {
                from.withdraw(amount);
                to.deposit(amount);
                System.out.println("Transferred " + amount + " from " + fromAccountNumber + " to " + toAccountNumber);
            } else {
                System.out.println("Transfer failed. Check amount or balance.");
            }
        }
    }
}

// 3. Main class
public class Main {
    public static void main(String[] args) {
        Bank bank = new Bank();

        // Create two accounts
        Account account1 = new Account("A001", "Alice", 1000);
        Account account2 = new Account("A002", "Bob", 500);

        // Add accounts to bank
        bank.addAccount(account1);
        bank.addAccount(account2);

        // Deposit and withdraw money
        account1.deposit(200);
        account2.withdraw(100);

        // Transfer money between accounts
        bank.transfer("A001", "A002", 300);

        // Print final balances
        System.out.println("\nFinal account balances:");
        System.out.println(account1);
        System.out.println(account2);
    }
}
"""
reference = """
import java.util.ArrayList;
import java.util.List;

class Account {
    private String accountNumber;
    private String owner;
    private double balance;

    public Account(String accountNumber, String owner, double balance) {
        this.accountNumber = accountNumber;
        this.owner = owner;
        this.balance = balance;
    }

    public void deposit(double amount) {
        balance += amount;
    }

    public boolean withdraw(double amount) {
        if (amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }

    public String getAccountNumber() {
        return accountNumber;
    }

    public double getBalance() {
        return balance;
    }

    @Override
    public String toString() {
        return owner + " (" + accountNumber + "): $" + balance;
    }
}

class Bank {
    private List<Account> accounts = new ArrayList<>();

    public void addAccount(Account account) {
        accounts.add(account);
    }

    public Account findAccount(String accountNumber) {
        for (Account a : accounts) {
            if (a.getAccountNumber().equals(accountNumber)) {
                return a;
            }
        }
        return null;
    }

    public boolean transfer(String fromAccount, String toAccount, double amount) {
        Account from = findAccount(fromAccount);
        Account to = findAccount(toAccount);
        if (from != null && to != null && from.withdraw(amount)) {
            to.deposit(amount);
            return true;
        }
        return false;
    }
}

public class Main {
    public static void main(String[] args) {
        Bank bank = new Bank();
        bank.addAccount(new Account("001", "Alice", 500));
        bank.addAccount(new Account("002", "Bob", 300));

        bank.findAccount("001").deposit(200);
        bank.findAccount("002").withdraw(50);
        bank.transfer("001", "002", 100);

        System.out.println(bank.findAccount("001"));
        System.out.println(bank.findAccount("002"));
    }
}
"""

result = calc_codebleu([reference], [prediction], lang="java", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)
# {
#   'codebleu': 0.5537, 
#   'ngram_match_score': 0.1041, 
#   'weighted_ngram_match_score': 0.1109, 
#   'syntax_match_score': 1.0, 
#   'dataflow_match_score': 1.0
# }