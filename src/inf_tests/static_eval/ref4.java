import java.util.ArrayList;
import java.util.List;

class Task {
    private int id;
    private String description;
    private boolean completed;

    public Task(int id, String description) {
        this.id = id;
        this.description = description;
        this.completed = false;
    }

    public void complete() {
        this.completed = true;
    }

    @Override
    public String toString() {
        return id + ": " + description + " [" + (completed ? "Done" : "Pending") + "]";
    }
}

class TaskManager {
    private List<Task> tasks = new ArrayList<>();

    public void addTask(Task t) {
        tasks.add(t);
    }

    public void markComplete(int id) {
        for (Task t : tasks) {
            if (t.id == id) {
                t.complete();
            }
        }
    }

    public List<Task> getTasks() {
        return tasks;
    }
}

public class Main {
    public static void main(String[] args) {
        TaskManager tm = new TaskManager();
        tm.addTask(new Task(1, "Write Code"));
        tm.addTask(new Task(2, "Review PR"));

        tm.markComplete(1);

        for (Task t : tm.getTasks()) {
            System.out.println(t);
        }
    }
}