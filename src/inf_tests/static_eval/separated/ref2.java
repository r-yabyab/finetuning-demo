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