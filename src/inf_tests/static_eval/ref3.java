import java.util.ArrayList;
import java.util.List;

class Student {
    private String id;
    private String name;
    private List<Double> grades = new ArrayList<>();

    public Student(String id, String name) {
        this.id = id;
        this.name = name;
    }

    public void addGrade(double grade) {
        grades.add(grade);
    }

    public double getAverage() {
        if (grades.isEmpty()) return 0;
        double sum = 0;
        for (double g : grades) sum += g;
        return sum / grades.size();
    }

    @Override
    public String toString() {
        return name + " (" + id + ") - Avg: " + getAverage();
    }
}

class School {
    private List<Student> students = new ArrayList<>();

    public void addStudent(Student s) {
        students.add(s);
    }

    public Student findStudent(String id) {
        for (Student s : students) {
            if (s.id.equals(id)) {
                return s;
            }
        }
        return null;
    }

    public List<Student> getStudents() {
        return students;
    }
}

public class Main {
    public static void main(String[] args) {
        School school = new School();
        Student s1 = new Student("S001", "John");
        Student s2 = new Student("S002", "Jane");

        s1.addGrade(90);
        s1.addGrade(85);
        s2.addGrade(78);
        s2.addGrade(88);

        school.addStudent(s1);
        school.addStudent(s2);

        for (Student s : school.getStudents()) {
            System.out.println(s);
        }
    }
}