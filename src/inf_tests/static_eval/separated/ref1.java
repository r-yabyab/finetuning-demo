import java.util.ArrayList;
import java.util.List;

class Book {
    private String title;
    private String author;
    private String isbn;

    public Book(String title, String author, String isbn) {
        this.title = title;
        this.author = author;
        this.isbn = isbn;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public String getIsbn() {
        return isbn;
    }

    @Override
    public String toString() {
        return title + " by " + author + " (ISBN: " + isbn + ")";
    }
}

class Library {
    private List<Book> books = new ArrayList<>();

    public void addBook(Book book) {
        books.add(book);
    }

    public boolean removeBookByIsbn(String isbn) {
        return books.removeIf(b -> b.getIsbn().equals(isbn));
    }

    public Book findBookByTitle(String title) {
        for (Book b : books) {
            if (b.getTitle().equalsIgnoreCase(title)) {
                return b;
            }
        }
        return null;
    }

    public List<Book> getBooks() {
        return books;
    }
}

public class Main {
    public static void main(String[] args) {
        Library library = new Library();
        library.addBook(new Book("1984", "George Orwell", "12345"));
        library.addBook(new Book("Brave New World", "Aldous Huxley", "67890"));

        Book found = library.findBookByTitle("1984");
        if (found != null) {
            System.out.println("Found: " + found);
        }

        library.removeBookByIsbn("12345");

        System.out.println("Remaining books:");
        for (Book b : library.getBooks()) {
            System.out.println(b);
        }
    }
}