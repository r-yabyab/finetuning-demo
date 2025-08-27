import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class ArrayProcessorTest {
    private ArrayProcessor processor;

    @BeforeEach
    void setUp() {
        processor = new ArrayProcessor();
    }

    @Test
    void testFindMax() {
        assertEquals(5, processor.findMax(new int[]{1, 3, 5, 2, 4}));
        assertEquals(10, processor.findMax(new int[]{10}));
        assertEquals(-1, processor.findMax(new int[]{-5, -3, -1, -4}));
        assertThrows(IllegalArgumentException.class, () -> processor.findMax(null));
        assertThrows(IllegalArgumentException.class, () -> processor.findMax(new int[]{}));
    }

    @Test
    void testFindMin() {
        assertEquals(1, processor.findMin(new int[]{1, 3, 5, 2, 4}));
        assertEquals(10, processor.findMin(new int[]{10}));
        assertEquals(-5, processor.findMin(new int[]{-5, -3, -1, -4}));
        assertThrows(IllegalArgumentException.class, () -> processor.findMin(null));
        assertThrows(IllegalArgumentException.class, () -> processor.findMin(new int[]{}));
    }

    @Test
    void testCalculateAverage() {
        assertEquals(3.0, processor.calculateAverage(new int[]{1, 2, 3, 4, 5}), 0.001);
        assertEquals(10.0, processor.calculateAverage(new int[]{10}), 0.001);
        assertEquals(2.5, processor.calculateAverage(new int[]{1, 2, 3, 4}), 0.001);
        assertThrows(IllegalArgumentException.class, () -> processor.calculateAverage(null));
        assertThrows(IllegalArgumentException.class, () -> processor.calculateAverage(new int[]{}));
    }

    @Test
    void testRemoveDuplicates() {
        assertArrayEquals(new int[]{1, 2, 3, 4}, processor.removeDuplicates(new int[]{1, 2, 2, 3, 3, 3, 4}));
        assertArrayEquals(new int[]{5}, processor.removeDuplicates(new int[]{5, 5, 5}));
        assertArrayEquals(new int[]{1, 2, 3}, processor.removeDuplicates(new int[]{1, 2, 3}));
        assertArrayEquals(new int[]{}, processor.removeDuplicates(new int[]{}));
        assertThrows(IllegalArgumentException.class, () -> processor.removeDuplicates(null));
    }
}
