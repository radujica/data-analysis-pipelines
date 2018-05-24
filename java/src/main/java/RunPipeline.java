import org.apache.commons.cli.*;

import java.io.IOException;

public class RunPipeline {
    public static void main(String... args) {
        Options options = new Options();
        Option pathOption = new Option("p", "path", true, "path to folder containing input files");
        pathOption.setRequired(true);
        options.addOption(pathOption);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            String path = cmd.getOptionValue("path");

            new Pipeline().start(path);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("Pipeline", options);
        }
    }
}
