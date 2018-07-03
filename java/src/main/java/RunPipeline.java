import org.apache.commons.cli.*;

import java.io.IOException;

public class RunPipeline {
    public static void main(String... args) {
        Options options = new Options();
        Option inputOption = new Option("i",
                "input",
                true,
                "Path to folder containing input files");
        inputOption.setRequired(true);
        options.addOption(inputOption);
        Option sliceOption = new Option("s",
                "slice",
                true,
                "Start and stop of a subset of the data");
        sliceOption.setRequired(true);
        options.addOption(sliceOption);
        Option outputOption = new Option("o",
                "output",
                true,
                "Path to output folder");
        outputOption.setRequired(true);
        options.addOption(outputOption);


        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            String input = cmd.getOptionValue("input");
            String slice = cmd.getOptionValue("slice");
            String output = cmd.getOptionValue("output");

            new Pipeline().start(input, slice, output);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("Pipeline", options);
        }
    }
}
