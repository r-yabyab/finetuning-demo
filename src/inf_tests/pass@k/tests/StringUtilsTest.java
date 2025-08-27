import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class StringUtilsTest {

    @Test
    void testReverseString() {
        StringUtils utils = new StringUtils();
        assertEquals("olleh", utils.reverseString("hello"));
        assertEquals("a", utils.reverseString("a"));
        assertEquals("", utils.reverseString(""));
        assertNull(utils.reverseString(null));
        assertEquals("!@#", utils.reverseString("#@!"));
    }

    // @Test
    // void testIsPalindrome() {
    //     assertTrue(utils.isPalindrome("racecar"));
    //     assertTrue(utils.isPalindrome("RaceCar"));
    //     assertTrue(utils.isPalindrome("a"));
    //     assertTrue(utils.isPalindrome(""));
    //     assertFalse(utils.isPalindrome("hello"));
    //     assertFalse(utils.isPalindrome(null));
    //     assertTrue(utils.isPalindrome("A man a plan a canal Panama"));
    // }

    // @Test
    // void testCountVowels() {
    //     assertEquals(2, utils.countVowels("hello"));
    //     assertEquals(5, utils.countVowels("aeiou"));
    //     assertEquals(5, utils.countVowels("AEIOU"));
    //     assertEquals(0, utils.countVowels("xyz"));
    //     assertEquals(0, utils.countVowels(""));
    //     assertEquals(0, utils.countVowels(null));
    //     assertEquals(3, utils.countVowels("Programming"));
    // }
}
