package ai.skymind.demo;

import ai.skymind.ApiClient;
import ai.skymind.ApiException;
import ai.skymind.Configuration;
import ai.skymind.skil.DefaultApi;
import ai.skymind.skil.model.INDArray;
import ai.skymind.skil.model.LoginRequest;
import ai.skymind.skil.model.LoginResponse;
import ai.skymind.skil.model.MultiPredictResponse;

import java.io.File;
import java.util.UUID;

public class Main {

    public static void main(String[] args) throws ApiException {
        String imagePath = args[0];
        File imageFile = new File(imagePath);

        if (!imageFile.exists()) {
            System.out.println("Specified file did not exist.");
            System.exit(1);
        }
        ApiClient defaultClient = Configuration.getDefaultApiClient();
        defaultClient.setBasePath("http://127.0.0.1:9008");

        // Configure API key authorization: api_key
        DefaultApi skil = new DefaultApi();
        LoginRequest loginRequest = new LoginRequest();
        loginRequest.setUserId("admin");
        loginRequest.setPassword("admin123");
        LoginResponse loginResponse = skil.login(loginRequest);

        String token = loginResponse.getToken();

        defaultClient.setApiKey(token);
        defaultClient.setApiKeyPrefix("Bearer");

        MultiPredictResponse multiPredictResponse = skil.multipredictimage(
                imageFile,
                UUID.randomUUID().toString(),
                false,
                "demo",
                "default",
                "wideresnet"
        );

        assert(multiPredictResponse.getOutputs().size() == 2);
        INDArray indArray1 = multiPredictResponse.getOutputs().get(0);
        INDArray indArray2 = multiPredictResponse.getOutputs().get(1);

        System.out.println(indArray1.getData().toString());
        System.out.println(indArray2.getData().toString());
    }
}
