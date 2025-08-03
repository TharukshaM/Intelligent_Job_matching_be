// at the top of generate_question_with_refs.js
import dotenv from "dotenv";
dotenv.config({ path: "../.env" }); // adjust path if needed

import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";

const token = process.env["GITHUB_TOKEN_MODULE_2"];
const endpoint = "https://models.github.ai/inference";
const model = "openai/gpt-4.1";

console.log("Token:", process.env["GITHUB_TOKEN_MODULE_2"]);


export async function main() {
  const algorithm = process.argv[2];
  const experience = process.argv[3];
  const language = process.argv[4];

  const client = ModelClient(endpoint, new AzureKeyCredential(token));

  const prompt = `
You are a programming interview assistant.

Generate a coding question for a ${experience}-level candidate that uses the ${algorithm} algorithm in ${language}.
Then provide two correct reference implementations in ${language}.

Format:
Q: <question>
A1: <reference solution 1>
A2: <reference solution 2>
`;

  const response = await client.path("/chat/completions").post({
    body: {
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      top_p: 1,
      model: model
    },
    contentType: "application/json"
  });

  if (isUnexpected(response)) {
    console.error("Error:", response.body);
    process.exit(1);
  }

  // ✅ THIS IS THE ACTUAL CONTENT
  const resultText = response.body.choices[0].message.content;
  console.log(resultText);
}

main().catch((err) => {
  console.error("Failed:", err);
});
