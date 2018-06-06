import org.apache.commons.cli.*;

import java.io.IOException;

public class RunPipeline {
    public static void main(String... args) {
        Options options = new Options();
        Option pathOption = new Option("i",
                "input",
                true,
                "Path to folder containing input files");
        pathOption.setRequired(true);
        options.addOption(pathOption);
        options.addOption(new Option("o",
                "output",
                true,
                "Path to output folder"));
        options.addOption(new Option("c",
                "check",
                false,
                "If passed, create output to check correctness of the pipeline, so output is saved '\n" +
                "                         'to csv files in --output folder. Otherwise, prints to stdout"));

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            String input = cmd.getOptionValue("input");
            String output = null;
            if (cmd.hasOption("check")) {
                if (!cmd.hasOption("output")) {
                    throw new RuntimeException("if checking, require --output arg");
                }

                output = cmd.getOptionValue("output");
            }

            new Pipeline().start(input, output);
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("Pipeline", options);
        }
    }
}
