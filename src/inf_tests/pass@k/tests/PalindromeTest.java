import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringUtilsTest {

    @Test
    void testIsPalindrome() {
        assertTrue(utils.isPalindrome("racecar"));
        assertTrue(utils.isPalindrome("RaceCar"));
        // assertTrue(utils.isPalindrome("a"));
        // assertTrue(utils.isPalindrome(""));
        // assertFalse(utils.isPalindrome("hello"));
        // assertFalse(utils.isPalindrome(null));
        // assertTrue(utils.isPalindrome("A man a plan a canal Panama"));
    }
}
