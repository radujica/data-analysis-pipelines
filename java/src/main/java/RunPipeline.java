import java.io.IOException;

public class RunPipeline {
    public static void main(String... args) {
        try {
            new Pipeline().start();
        } catch (IOException e) {
            System.out.println("NOOOOOO");
        }
    }
}
