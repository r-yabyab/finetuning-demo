import java.util.ArrayList;
import java.util.List;

class Item {
    private String name;
    private int quantity;
    private double price;

    public Item(String name, int quantity, double price) {
        this.name = name;
        this.quantity = quantity;
        this.price = price;
    }

    public double getTotalValue() {
        return quantity * price;
    }

    @Override
    public String toString() {
        return name + " x" + quantity + " @ $" + price;
    }
}

class Inventory {
    private List<Item> items = new ArrayList<>();

    public void addItem(Item item) {
        items.add(item);
    }

    public boolean removeItem(String name) {
        return items.removeIf(i -> i.name.equals(name));
    }

    public double getTotalInventoryValue() {
        double total = 0;
        for (Item i : items) total += i.getTotalValue();
        return total;
    }

    public List<Item> getItems() {
        return items;
    }
}

public class Main {
    public static void main(String[] args) {
        Inventory inv = new Inventory();
        inv.addItem(new Item("Laptop", 2, 1000));
        inv.addItem(new Item("Mouse", 5, 25));

        inv.removeItem("Mouse");

        System.out.println("Inventory:");
        for (Item i : inv.getItems()) {
            System.out.println(i);
        }
        System.out.println("Total value: $" + inv.getTotalInventoryValue());
    }
}